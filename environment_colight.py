import gym
import numpy as np
import cityflow

class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.

    Parameters
    ----------
    world: World object
    agents: list of agent, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """
    def __init__(self, world, agents, metric):
        self.world = world
        
        self.eng = self.world.eng
        self.n_agents = len(self.world.intersection_ids)
        self.n = self.n_agents

        # assert len(agents) == self.n_agents # comment it when parameter sharing

        self.agents = agents
        if isinstance(self.agents,list):
            action_dims = [agent.action_space.n for agent in agents]
            self.action_space = gym.spaces.MultiDiscrete(action_dims)
        else:
            single_action = [len(world.intersections[0].phases)]
            action_dim = single_action*self.n_agents
            self.action_space = gym.spaces.MultiDiscrete(action_dim)
        self.metric = metric

    def step(self, actions):
        assert len(actions) == self.n_agents

        self.world.step(actions)

        if isinstance(self.agents,list):
            obs = [agent.get_ob() for agent in self.agents]
            rewards = [agent.get_reward() for agent in self.agents]
        else:
            obs = self.agents.get_ob()
            rewards = self.agents.get_reward()
        dones = [False] * self.n_agents
        #infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        if isinstance(self.agents,list):
            obs = [agent.get_ob() for agent in self.agents]
        else:
            obs = self.agents.get_ob()
        return obs
