import asyncio
from typing import Any, Dict, List

from datetime import datetime as dt
import datetime

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
import json


def to_serializable(obj):
    if isinstance(obj, datetime.datetime):   # ✅ 模块名.类型名
        return obj.isoformat()
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
    trigger_news: Dict={}
    # tweet_db(firehose): store the tweets of all users; key: tweet_id, value: message
    tweet_db = {}
    output_path=""
    target="Metoo Movement"
    abm_model:mesa.Model = None
    opinion_history: Dict = Field(default_factory=dict)
    class Config:
        arbitrary_types_allowed = True
    # @validator("time_delta")
    # def convert_str_to_timedelta(cls, string):
    #
    #     return datetime.timedelta(seconds=int(string))

    def __init__(self, rule, **kwargs):#rule是env_config中的一部分，**kwargs是env_config中除去rule的剩余部分
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
        self.rule.update_visible_agents(self)#让环境更新 每个 Agent 当前能“看到”的其它 agent

    async def step(self) -> List[Message]:
        """Run one step of the environment"""

        logger.info(f"Tick tock. Current time: {self.current_time}")

        # Get the next agent index
        agent_ids = self.rule.get_next_agent_idx(self)

        # Get the personal experience of each agent
        await asyncio.gather(
                    *[
                        self.agents[i].get_personal_experience()
                        for i in agent_ids
                    ]
        )   

        # Generate current environment description
        env_descriptions = self.rule.get_env_description(self)

        # check whether the news is a tweet; if so, add to the tweet_db
        self.check_tweet(env_descriptions)
        env_descriptions = self.rule.get_env_description(self)

        # Generate the next message
        messages = await asyncio.gather(
            *[
                self.agents[i].astep(self.current_time, env_descriptions[i])
                for i in agent_ids
            ]
        )

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

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns

    def tick_tock(self) -> None:
        """Increment the time"""
        self.current_time = self.current_time + datetime.timedelta(
            seconds=self.time_delta
        )


    # def save_data_collector(self) -> None:
    #     """Output the data collector to the target file"""
    #     data = {}
    #     for agent in self.agents:
    #         data[agent.name] = agent.data_collector
    #     # naive agents in ABM model
    #     if self.abm_model is not None:
    #         opinion = {}
    #         for agent in self.abm_model.schedule.agents:
    #             opinion[agent.name] = agent.att[-1]
    #         data['opinion_results'] = opinion
    #         # with open(r'E:\Mxy\HiSim\temp\att.json', 'w', encoding='utf-8') as w:
    #         #     json.dump(
    #         #         opinion,
    #         #         w,
    #         #         ensure_ascii=False,
    #         #         indent=2,
    #         #         default=to_serializable  # ✅ 关键就在这里
    #         #     )
    #     print('Output to {}'.format(self.output_path))
    #     with open(self.output_path,'wb') as f:
    #         pickle.dump(data, f)
    #     # ✅ 2. 额外保存为 txt（可读文本）
    #     with open(r'E:\Mxy\Hisim\temp\action.json', 'w', encoding='utf-8') as w:
    #         json.dump(
    #             data,
    #             w,
    #             ensure_ascii=False,
    #             indent=2,
    #             default=to_serializable  # ✅ 关键就在这里
    #         )
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

            # 初始化opinion_results如果不存在
            if not hasattr(self, 'opinion_history'):
                self.opinion_history = {}

            # 使用当前保存次数作为键
            save_count = len(self.opinion_history)
            self.opinion_history[str(save_count)] = opinion

            # 将历史记录保存到data中
            data['opinion_results'] = self.opinion_history

            # 如果你只想保存当前这次的结果（而不需要历史），使用以下代码：
            # data['opinion_results'] = {str(save_count): opinion}

            # with open(r'E:\Mxy\HiSim\temp\att.json', 'w', encoding='utf-8') as w:
            #     json.dump(
            #         opinion,
            #         w,
            #         ensure_ascii=False,
            #         indent=2,
            #         default=to_serializable  # ✅ 关键就在这里
            #     )

        print('Output to {}'.format(self.output_path))
        with open(self.output_path, 'wb') as f:
            pickle.dump(data, f)
        output_path = self.output_path

        # 1️⃣ 按 / 分割
        parts = output_path.split("/")

        # 2️⃣ 获取前面的路径
        base_path = "/".join(parts[:-1])

        # 3️⃣ 获取文件名并改后缀为 .json
        filename = parts[-1].split(".")[0] + ".json"

        # 4️⃣ 拼接 temp
        final_path = f"{base_path}/temp/{filename}"

        # 5️⃣ 确保 temp 目录存在
        import os
        os.makedirs(f"{base_path}/temp", exist_ok=True)

        # ✅ 最终写入
        with open(final_path, 'w', encoding='utf-8') as w:
            json.dump(
                data,
                w,
                ensure_ascii=False,
                indent=2,
                default=to_serializable
            )

    def check_tweet(self, env_descptions):
        cnt_turn = self.cnt_turn
        if 'posts a tweet' in env_descptions[0]:
            author = env_descptions[0][:env_descptions[0].index('posts a tweet')].strip()
            content = env_descptions[0]
            msg_lst = self.rule.update_tweet_db_for_news(self, author, content)
            self.rule.update_tweet_page_for_news(self, msg_lst)
            # del the trigger news
            self.trigger_news[cnt_turn]=""

    async def test(self, agent, context) -> List[Message]:
        """Run one step of the environment"""
        """Test the system from micro-level"""

        # Generate the next message
        prompt, message, parsed_response = await agent.acontext_test(context)

        return prompt, message, parsed_response