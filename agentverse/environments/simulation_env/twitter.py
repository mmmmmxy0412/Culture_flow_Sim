import asyncio
from typing import Any, Dict, List

from datetime import datetime as dt
import datetime
import json

from pydantic import Field

from agentverse.logging import logger
from agentverse.environments import env_registry as EnvironmentRegistry
from agentverse.agents.simulation_agent.conversation import BaseAgent

from agentverse.environments.simulation_env.rules.twitter import TwitterRule as Rule
from agentverse.message import Message

from ..base import BaseEnvironment

from pydantic import validator

import pickle
import mesa
import os
import re


def to_serializable(obj):
    """将不可序列化的对象转换为可序列化的形式"""
    if isinstance(obj, dt):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


@EnvironmentRegistry.register("twitter")
class TwitterEnvironment(BaseEnvironment):
    """
    Environment used in Observation-Planning-Reflection agent architecture.

    Args:
        agents: List of agents
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
        current_time
        time_delta: time difference between steps
        trigger_news: Dict, time(turn index) and desc of emergent events
    """

    agents: List[BaseAgent]
    rule: Rule
    max_turns: int = 10
    cnt_turn: int = 0
    last_messages: List[Message] = []
    rule_params: Dict = {}
    current_time: dt = dt.now()
    time_delta: int = 120
    trigger_news: Dict = {}
    # tweet_db(firehose): store the tweets of all users; key: tweet_id, value: message
    tweet_db = {}
    output_path: str = ""
    target: str = "Metoo Movement"
    abm_model: mesa.Model = None

    # 新增：用于记录所有 agent 的发言
    agent_responses_history: Dict = Field(default_factory=dict)
    json_output_path: str = ""
    opinion_history: Dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, rule, **kwargs):
        rule_config = rule
        order_config = rule_config.get("order", {"type": "sequential"})
        visibility_config = rule_config.get("visibility", {"type": "all"})
        selector_config = rule_config.get("selector", {"type": "basic"})
        updater_config = rule_config.get("updater", {"type": "basic"})
        describer_config = rule_config.get("describer", {"type": "basic"})
        rule = Rule(
            order_config,
            visibility_config,
            selector_config,
            updater_config,
            describer_config,
        )

        super().__init__(rule=rule, **kwargs)
        self.rule.update_visible_agents(self)

        # 初始化 JSON 输出路径
        self._init_json_output_path()

        # 初始化历史记录
        self.agent_responses_history = {}
        self.opinion_history = {}

    def _init_json_output_path(self):
        """初始化 JSON 输出路径"""
        if self.output_path:
            # 按 / 分割
            parts = self.output_path.split("/")
            # 获取前面的路径
            base_path = "/".join(parts[:-1])
            # 获取文件名并改后缀为 .json
            filename = parts[-1].split(".")[0] + ".json"
            # 拼接 temp
            self.json_output_path = f"{base_path}/temp/{filename}"
            # 确保 temp 目录存在
            os.makedirs(f"{base_path}/temp", exist_ok=True)
        else:
            logger.warning("output_path not set, cannot initialize json_output_path")

    async def step(self) -> List[Message]:
        """Run one step of the environment"""

        logger.info(f"Tick tock. Current time: {self.current_time}")

        # Get the next agent index
        agent_ids = self.rule.get_next_agent_idx(self)

        # Get the personal experience of each agent (流式处理)
        await self._gather_personal_experiences(agent_ids)

        # Generate current environment description
        env_descriptions = self.rule.get_env_description(self)

        # check whether the news is a tweet; if so, add to the tweet_db
        self.check_tweet(env_descriptions)
        env_descriptions = self.rule.get_env_description(self)

        # Generate the next message (流式处理)
        messages = await self._gather_agent_steps(agent_ids, env_descriptions)

        # Some rules will select certain messages from all the messages
        selected_messages = self.rule.select_message(self, messages)
        self.last_messages = selected_messages
        self.print_messages(selected_messages)

        # Update opinion of mirror and other naive agents
        # update naive agents
        if self.abm_model is not None:
            self.abm_model.step()
            # then substitude the value of mirror using LLM results
            for i in agent_ids:
                self.abm_model.update_mirror(self.agents[i].name, self.agents[i].atts[-1])

        # Update the database of public tweets
        self.rule.update_tweet_db(self)
        print('Tweet Database Updated.')

        # Update the memory of the agents
        self.rule.update_memory(self)
        print('Agent Memory Updated.')

        # Update tweet page of agents
        self.rule.update_tweet_page(self)
        print('Tweet Pages Updated.')

        # TODO: Update the notifications(info box) of agents
        self.rule.update_info_box(self)
        print('Tweet Infobox Updated.')

        # Update the set of visible agents for each agent
        self.rule.update_visible_agents(self)
        print('Visible Agents Updated.')

        self.cnt_turn += 1

        # update current_time
        self.tick_tock()

        return selected_messages

    async def _gather_personal_experiences(self, agent_ids: List[int]) -> None:
        """
        使用 asyncio.gather 逐个处理 agent 的个人经历
        每个 agent 完成后立即打印日志，无需等待所有 agent
        """

        # 为每个 agent 创建一个包装的 coroutine，在完成时立即打印
        async def get_experience_and_log(agent_idx):
            try:
                await self.agents[agent_idx].get_personal_experience()
                logger.info(f"Agent {self.agents[agent_idx].name} personal experience gathered")
            except Exception as e:
                logger.error(f"Error gathering personal experience for agent {agent_idx}: {e}")

        # 并发执行所有 agent
        await asyncio.gather(
            *[get_experience_and_log(i) for i in agent_ids],
            return_exceptions=True
        )

    async def _gather_agent_steps(self, agent_ids: List[int], env_descriptions: List[str]) -> List[Message]:
        """
        使用 asyncio.gather 逐个处理 agent 的回答
        每个 agent 完成后立即保存其内容到 JSON，无需等待所有 agent
        """

        # 为每个 agent 创建一个包装的 coroutine，在完成时立即保存
        async def step_and_save(agent_idx):
            try:
                message = await self.agents[agent_idx].astep(self.current_time, env_descriptions[agent_idx])
                if message is not None:
                    logger.info(f"{message.sender}: {message.content}")

                    # ✅ 立即保存该 agent 的回答（包含 thought、action、attitude）
                    self._save_agent_response(
                        agent=self.agents[agent_idx],
                        message=message,
                        timestamp=self.current_time,
                        turn=self.cnt_turn
                    )
                return message
            except Exception as e:
                logger.error(f"Error in agent {agent_idx} step: {e}")
                return None

        # 并发执行所有 agent
        messages = await asyncio.gather(
            *[step_and_save(i) for i in agent_ids],
            return_exceptions=True
        )

        # 确保返回的是 Message 对象而不是异常
        return [msg if isinstance(msg, Message) else None for msg in messages]

    # def _extract_thought_and_action(self, agent) -> tuple:
    #     """
    #     从 agent 的 data_collector 中提取 thought 和 action
    #
    #     Returns:
    #         (thought, action) - 从最新的轮次中提取
    #     """
    #     try:
    #         turn = self.cnt_turn
    #         if turn in agent.data_collector:
    #             collector = agent.data_collector[turn]
    #
    #             # 提取 Thought
    #             thought = ""
    #             if 'prompt' in collector:
    #                 prompt = collector['prompt']
    #                 # 从 prompt 中提取 thought（通常在 Thought: 之后）
    #                 thought_match = re.search(r'Thought:\s*(.+?)(?:\n|$)', prompt)
    #                 if thought_match:
    #                     thought = thought_match.group(1).strip()
    #
    #             # 提取 Action
    #             action = ""
    #             if 'parsed_response' in collector:
    #                 parsed_response = collector['parsed_response']
    #
    #                 # 1) 如果是 dict 且有 output，直接用（你原来就写了这个分支）
    #                 if isinstance(parsed_response, dict) and 'output' in parsed_response:
    #                     action_text = parsed_response['output']  # e.g. "do_nothing()"
    #
    #                 # 2) 如果是 AgentFinish 这类对象，通常会有 return_values 属性
    #                 elif hasattr(parsed_response, "return_values") and isinstance(parsed_response.return_values, dict):
    #                     action_text = parsed_response.return_values.get("output", "")
    #
    #                 else:
    #                     action_text = str(parsed_response)
    #
    #                 # 你只想要 do_nothing 而不是 do_nothing()
    #                 m = re.search(r'([a-zA-Z_]\w*)\s*\(', action_text)
    #                 action = m.group(1) if m else action_text.strip()
    #
    #             # 提取 Attitude
    #             attitude = collector.get('att', 0)
    #
    #             return thought, action, attitude
    #     except Exception as e:
    #         logger.error(f"Error extracting thought and action: {e}")
    #
    #     return "", "", 0

    def _extract_thought_and_action(self, agent) -> tuple:
        """
        从 agent 的 data_collector 中提取 thought 和 action

        Returns:
            (thought, action, attitude) - 从最新的轮次中提取
        """
        try:
            turn = self.cnt_turn
            if turn in agent.data_collector:
                collector = agent.data_collector[turn]

                # ===================== 提取 Thought =====================
                thought = ""

                # 1) 优先从 parsed_response 的 log 里提取 Thought
                if 'parsed_response' in collector:
                    parsed_response = collector['parsed_response']
                    log_text = ""

                    # a) dict：可能有 log 字段
                    if isinstance(parsed_response, dict) and 'log' in parsed_response:
                        log_text = parsed_response.get('log', "") or ""

                    # b) AgentFinish / 对象：通常有 .log 属性
                    elif hasattr(parsed_response, "log"):
                        log_text = getattr(parsed_response, "log", "") or ""

                    # c) 兜底：str(parsed_response) 里用正则抠 log='...'
                    else:
                        s = str(parsed_response)
                        log_match = re.search(r"log='([\s\S]*?)'\s*\)?$", s)
                        if log_match:
                            log_text = log_match.group(1)

                    if log_text:
                        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", log_text)
                        if thought_match:
                            thought = thought_match.group(1).strip()

                # 2) 如果 parsed_response 没提到 thought，再从 prompt 里提取（保留你原逻辑）
                if not thought and 'prompt' in collector:
                    prompt = collector['prompt']
                    thought_match = re.search(r'Thought:\s*(.+?)(?:\n|$)', prompt)
                    if thought_match:
                        thought = thought_match.group(1).strip()

                # ===================== 提取 Action（保持不变）=====================
                action = ""
                if 'parsed_response' in collector:
                    parsed_response = collector['parsed_response']

                    # 1) 如果是 dict 且有 output，直接用（你原来就写了这个分支）
                    if isinstance(parsed_response, dict) and 'output' in parsed_response:
                        action_text = parsed_response['output']  # e.g. "do_nothing()"

                    # 2) 如果是 AgentFinish 这类对象，通常会有 return_values 属性
                    elif hasattr(parsed_response, "return_values") and isinstance(parsed_response.return_values, dict):
                        action_text = parsed_response.return_values.get("output", "")

                    else:
                        action_text = str(parsed_response)

                    # 你只想要 do_nothing 而不是 do_nothing()
                    m = re.search(r'([a-zA-Z_]\w*)\s*\(', action_text)
                    action = m.group(1) if m else action_text.strip()

                # ===================== 提取 Attitude（只取最新一轮）=====================
                attitude = 0
                att_data = collector.get('att', 0)

                if isinstance(att_data, list):
                    # 如果是累计列表，只取最后一个
                    attitude = att_data[-1] if att_data else 0

                elif isinstance(att_data, tuple):
                    attitude = att_data[-1] if att_data else 0

                elif isinstance(att_data, dict):
                    # 如果未来 att 被存成按 turn 组织的字典，优先取当前 turn
                    if turn in att_data:
                        attitude = att_data[turn]
                    elif str(turn) in att_data:
                        attitude = att_data[str(turn)]
                    else:
                        # 兜底：取最后一个 value
                        try:
                            attitude = list(att_data.values())[-1]
                        except Exception:
                            attitude = 0

                else:
                    # 单值直接用
                    attitude = att_data

                return thought, action, attitude
        except Exception as e:
            logger.error(f"Error extracting thought and action: {e}")

        return "", "", 0
    def _save_agent_response(self, agent, message: Message, timestamp: dt, turn: int) -> None:
        """
        保存单个 agent 的回答到 JSON 文件
        包含：thought、action、attitude、content

        Args:
            agent: agent 对象
            message: agent 的 Message 对象
            timestamp: 发言时间
            turn: 当前轮数
        """
        try:
            # 从 agent 的 data_collector 中提取 thought、action 和 attitude
            thought, action, attitude = self._extract_thought_and_action(agent)

            # 初始化当前 turn 的数据结构
            turn_key = f"turn_{turn}"
            if turn_key not in self.agent_responses_history:
                self.agent_responses_history[turn_key] = {
                    "timestamp": timestamp.isoformat() if isinstance(timestamp, dt) else str(timestamp),
                    "agents": {}
                }

            # 保存 agent 的完整信息
            self.agent_responses_history[turn_key]["agents"][message.sender] = {
                "thought": thought,  # ✅ Agent 的思考过程
                "action": action,  # ✅ Agent 的动作/反应
                "attitude": attitude,  # ✅ Agent 的态度值
                "content": message.content,  # Agent 的发言内容
                "saved_at": dt.now().isoformat()
            }

            # 立即写入 JSON 文件
            self._write_json_file()

            logger.debug(f"Agent {message.sender}'s response saved to JSON (thought, action, attitude)")

        except Exception as e:
            logger.error(f"Error saving agent response: {e}")

    def _write_json_file(self) -> None:
        """
        将当前的 agent_responses_history 写入 JSON 文件
        """
        if not self.json_output_path:
            logger.warning("json_output_path not set, cannot write JSON file")
            return

        try:
            with open(self.json_output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    self.agent_responses_history,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=to_serializable
                )
        except Exception as e:
            logger.error(f"Error writing JSON file: {e}")

    def print_messages(self, messages: List[Message]) -> None:
        for message in messages:
            if message is not None:
                logger.info(f"{message.sender}: {message.content}")

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()
        BaseAgent.update_forward_refs()
        for agent in self.agents:
            agent.reset(environment=self)

        # 重置历史记录
        self.agent_responses_history = {}
        self.opinion_history = {}

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns

    def tick_tock(self) -> None:
        """Increment the time"""
        self.current_time = self.current_time + datetime.timedelta(
            seconds=self.time_delta
        )

    def save_data_collector(self) -> None:
        """Output the data collector to the target file"""
        data = {}
        for agent in self.agents:
            data[agent.name] = agent.data_collector

        # naive agents in ABM model
        if self.abm_model is not None:
            opinion = {}
            for agent in self.abm_model.schedule.agents:
                opinion[agent.name] = agent.att[-1]

            # 使用当前保存次数作为键
            save_count = len(self.opinion_history)
            self.opinion_history[str(save_count)] = opinion

            # 将历史记录保存到data中
            data['opinion_results'] = self.opinion_history

        print('Output to {}'.format(self.output_path))
        with open(self.output_path, 'wb') as f:
            pickle.dump(data, f)

    def check_tweet(self, env_descptions):
        cnt_turn = self.cnt_turn
        if 'posts a tweet' in env_descptions[0]:
            author = env_descptions[0][:env_descptions[0].index('posts a tweet')].strip()
            content = env_descptions[0]
            msg_lst = self.rule.update_tweet_db_for_news(self, author, content)
            self.rule.update_tweet_page_for_news(self, msg_lst)
            # del the trigger news
            self.trigger_news[cnt_turn] = ""

    async def test(self, agent, context) -> List[Message]:
        """Run one step of the environment"""
        """Test the system from micro-level"""

        # Generate the next message
        prompt, message, parsed_response = await agent.acontext_test(context)

        return prompt, message, parsed_response