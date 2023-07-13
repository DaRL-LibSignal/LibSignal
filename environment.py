import gym
import pettingzoo
from pettingzoo.utils import agent_selector, wrappers
import numpy as np


class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.
    Parameters
    ----------
    world: World object
    agents: list of agents, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """

    def __init__(self, world, agents, metric):
        """
        :param world: one world object to interact with agents. Support multi world
        objects in different TSCEnvs.
        :param agents: single agents, each control all intersections. Or multi agents,
        each control one intersection.
        actions is a list of actions, agents is a list of agents.
        :param metric: metrics to evaluate policy.
        """
        self.world = world
        self.eng = self.world.eng
        self.n_agents = len(agents) * agents[0].sub_agents
        # test agents number == intersection number
        assert len(world.intersection_ids) == self.n_agents
        self.agents = agents
        action_dims = [agent.action_space.n * agent.sub_agents for agent in agents]
        # total action space of all agents.
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        self.metric = metric

    def step(self, actions):
        """
        :param actions: keep action as N_agents * 1
        """
        if not actions.shape:
            assert(self.n_agents == 1)
            actions = actions[np.newaxis]
        else:
            assert len(actions) == self.n_agents
        self.world.step(actions)

        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]
            # obs = np.expand_dims(np.array(obs),axis=1)
            rewards = [agent.get_reward() for agent in self.agents]
            # rewards = np.expand_dims(np.array(rewards),axis=1)
        else:
            obs = [self.agents[0].get_ob()]
            rewards = [self.agents[0].get_reward()]
        dones = [False] * self.n_agents
        # infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]  # [agent, sub_agent==1, feature]
            # obs = np.expand_dims(np.array(obs),axis=1)
        else:
            obs = [self.agents[0].get_ob()]  # [agent==1, sub_agent, feature]
        return obs


class TSCMAEnv(pettingzoo.AECEnv):
    """
    Environment for Traffic Signal Control task.
    Parameters
    ----------
    world: World object
    agents: list of agents, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """

    metadata = {"render.modes": ["rgb_array"], "name": "MARL_env mode"}
    def __init__(self, world, agents, metric, render_mode=None):
        """
        :param world: one world object to interact with agents. Support multi world
        objects in different TSCEnvs.
        :param agents: single agents, each control all intersections. Or multi agents,
        each control one intersection.
        actions is a list of actions, agents is a list of agents.
        :param metric: metrics to evaluate policy.
        """

        # TODO: multiagent mode need to clearly defined where those obs, rewards comes from
        self.world = world
        self.eng = self.world.eng
        # TODO: change from list of obj to list of name

        agents.sort(key=lambda x: x.rank)
        self.agent_objs = agents
        self.possible_agents = [a.id for a in self.agent_objs]
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # TODO: change subagent into agent -> compatibe to MARL Env
        # assign action_spaces for each agent
        self._action_spaces = {a.id: a.action_space for a in self.agent_objs}
        self._observation_spaces = {a.id: a.observation_space for a in self.agent_objs}
        # total action space of all agents.
        self.render_mode = render_mode
        self.metric = metric

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}

    def action_space(self, agent_id):
        return self._action_spaces[agent_id]
    
    def observation_space(self, agent_id):
        return self._observation_spaces[agent_id]
    
    # def render(self):
    #     """
    #     If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
    #     """

    #     self.render_mode == "rgb_array":
    #     # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
    #     #                          f"temp/img{self.sim_step}.jpg",
    #     #                          width=self.virtual_display[0],
    #     #                          height=self.virtual_display[1])
    #     img = self.disp.grab()
    #     return np.array(img)
    
    # lru_cache can reduce this fixed info retrival. but it will cause trouble to gc

    def step(self, action):
        """
        :param actions: keep action for each agent
        """
        # TODO: add step by step update later for heiarhical RL
        # self.agents order == self.agent_obj.rank order == self.word.intersections order == input action order

        agent = self.agent_selection
        



        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]
            # obs = np.expand_dims(np.array(obs),axis=1)
            rewards = [agent.get_reward() for agent in self.agents]
            # rewards = np.expand_dims(np.array(rewards),axis=1)
        else:
            obs = [self.agents[0].get_ob()]
            rewards = [self.agents[0].get_reward()]
        dones = [False] * self.n_agents
        # infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.world.reset()

        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]  # [agent, sub_agent==1, feature]
            # obs = np.expand_dims(np.array(obs),axis=1)
        return obs
    
