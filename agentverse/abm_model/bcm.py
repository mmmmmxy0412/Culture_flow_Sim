import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt
from agentverse.abm_model import abm_registry
from agentverse.logging import get_logger

logger = get_logger()


class BCAgent(mesa.Agent):
    """
    Deffuant's model is a BC model where N_J=1
    It assumes that if the message 𝑚𝑗,𝑡 is close enough to the agent 𝑖’s attitude 𝑎𝑖,𝑡 , 
    the message has an assimilation force on the agent’s attitude. 
    """

    def __init__(self, model, unique_id, name, init_att, alpha=0.5, bc_bound = 0.2):
        """
        bc_bound: the confidence bound
        """
        
        super().__init__(unique_id, model)
        
        self.name = name
        # initial attitude
        self.att =  [init_att]
        # strength of the social influence
        self.alpha = alpha
        self.bc_bound = bc_bound
        

    def step(self):
        """
        Selection Function: one random agent j in the system within the confidence bound
        Message Function: m_jt = a_jt
        Assimilation Force: asm(a_it, m_jt) = (m_jt-a_it)
        Similarity Bias: sim(a_it, m_jt) = 1 if diff< bc_bound, else 0
        """
        # attitude update
        att = self.att[-1]
        att_update = 0
        candidate_agents = []
        for agent in self.model.schedule.agents:
            # exclude the agent itself
            if agent == self:continue
            if abs(att-agent.att[-1])<self.bc_bound:
                candidate_agents.append(agent)
        influencer_names = []  # 新增：记录本轮影响当前 agent 的 id 列表
        # randomly sample
        if len(candidate_agents):
            target_agent = random.choice(candidate_agents)
            sim = 1
            att_update = target_agent.att[-1]-att
            influencer_names.append(target_agent.name)
        else:
            sim = 0 
            att_update = 0
        att = att + self.alpha * att_update
        self.att.append(att)
#         print(self.name, att)

        # 记录：以 name 为键值
        step_t = self.model._steps
        if step_t not in self.model.influence_log:
            self.model.influence_log[step_t] = {}
        self.model.influence_log[step_t][self.name] = {
            "att": att,
            "influencers": influencer_names,
        }


@abm_registry.register("bcm")
class BCModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, agent_config_lst, order = 'concurrent', alpha=0.1, bc_bound=0.1,llm_agents_atts=[]):
        super().__init__()
        self.num_agents = len(agent_config_lst)
        self.llm_agents_atts = llm_agents_atts
        self.name2idx = {}
        # Create scheduler and assign it to the model
        if order =='concurrent':
            self.schedule = mesa.time.BaseScheduler(self)
        elif order =='simultaneous':
            self.schedule = mesa.time.SimultaneousActivation(self)
        elif order =='random':
            self.schedule = mesa.time.RandomActivation(self)
        else:
            raise NotImplementedError

        # Create agents
        self.name2idx = {}
        for i in range(self.num_agents):
            a = BCAgent(self, agent_config_lst[i]['id'], agent_config_lst[i]['name'], agent_config_lst[i]['init_att'],
                         alpha=alpha, bc_bound = bc_bound)
            # Add the agent to the scheduler
            self.schedule.add(a)
            self.name2idx[agent_config_lst[i]['name']] = i
        assert list(self.name2idx.keys()) == [a.name for a in self.agents]
        assert self.num_agents == len(self.agents)
        self.influence_log = {}  # 内存中的日志
        self._influence_log_initialized = False  # 是否已经创建过文件（用于判断“第一轮”）

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()
        for agent in self.llm_agents_atts:
            self.update_mirror(agent, self.llm_agents_atts[agent][self._steps-1])
        self.save_influence_log(r"E:/Mxy/HiSim/influence_bcm/influence_bc_model.json")
        
    def get_attitudes(self):
        atts = [a.att[-1] for a in self.agents]
        return atts
    
    def get_measures(self, target_attitudes,ne_att=0):
        """
        target_attitudes: empirical data
        output measures: bias, diversity
        - bias: the deviation of the mean attitude from the neutral attitude
        - diversity: the standard deviation of attitudes
        """
        simu_atts = self.get_attitudes()
        
        # empirical
        bias = np.mean(target_attitudes)-ne_att
        diversity = np.var(target_attitudes)
        
        # simu
        simu_bias = np.mean(simu_atts)-ne_att
        simu_diversity = np.var(simu_atts)
        
        delta_bias = abs(simu_bias-bias)
        delta_diversity = abs(simu_diversity-diversity)

        return {'bias':bias,
               'diversity':diversity,
               'simu_bias':simu_bias,
               'simu_diversity':simu_diversity,
               'delta_bias':delta_bias,
               'delta_diversity':delta_diversity}

    def update_mirror(self, name, att):
        if name not in self.name2idx:
            raise KeyError(f"name '{name}' not in name2idx; available: {list(self.name2idx.keys())[:10]}")
        idx = self.name2idx[name]
        self.agents[idx].att[-1] = att

    def save_influence_log(self, path: str):
        import json, os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 当前内存里的日志
        current_log = self.influence_log

        # 第一次保存：直接写 current_log，覆盖
        if not self._influence_log_initialized or not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(current_log, f, ensure_ascii=False, indent=2)
            self._influence_log_initialized = True
            return

        # 之后每次：先读旧文件，再和 current_log 合并，然后写回
        try:
            with open(path, "r", encoding="utf-8") as f:
                old_log = json.load(f)
        except Exception:
            # 如果旧文件坏了或不能解析，就退化成覆盖
            old_log = {}

        # 合并：以 step 为 key，把内存里有的都写进去（覆盖同 step，保留其他 step）
        for step_str, data in current_log.items():
            # 注意 step_t 可能是 int，这里统一用 str，和 JSON 的 key 对齐
            s = str(step_str)
            old_log[s] = data

        with open(path, "w", encoding="utf-8") as f:
            json.dump(old_log, f, ensure_ascii=False, indent=2)
        self._influence_log_initialized = True