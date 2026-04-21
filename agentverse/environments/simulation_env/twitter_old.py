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
    output_path = ""
    target = "Metoo Movement"
    abm_model: mesa.Model = None

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
        每个 agent 完成后立即打印其内容，无需等待所有 agent
        """

        # 为每个 agent 创建一个包装的 coroutine，在完成时立即打印
        async def step_and_log(agent_idx):
            try:
                message = await self.agents[agent_idx].astep(self.current_time, env_descriptions[agent_idx])
                if message is not None:
                    logger.info(f"{message.sender}: {message.content}")
                return message
            except Exception as e:
                logger.error(f"Error in agent {agent_idx} step: {e}")
                return None

        # 并发执行所有 agent
        messages = await asyncio.gather(
            *[step_and_log(i) for i in agent_ids],
            return_exceptions=True
        )

        # 确保返回的是 Message 对象而不是异常
        return [msg if isinstance(msg, Message) else None for msg in messages]

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
            data['opinion_results'] = opinion
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