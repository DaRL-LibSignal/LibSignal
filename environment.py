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
        self.n_agents = len(agents)
        self.n = self.n_agents

        assert len(agents) == self.n_agents

        self.agents = agents
        action_dims = [agent.action_space.n for agent in agents]
        self.action_space = gym.spaces.MultiDiscrete(action_dims)

        self.metric = metric

    def change_world(self, world):
        self.world.reset()
        # self.world = world
        self.eng = world.eng
        self.world.reset()

    def step(self, actions):
        assert len(actions) == self.n_agents

        self.world.step(actions)

        obs = [agent.get_ob() for agent in self.agents]
        rewards = [agent.get_reward() for agent in self.agents]
        dones = [False] * self.n_agents
        # infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        obs = [agent.get_ob() for agent in self.agents]
        return obs
