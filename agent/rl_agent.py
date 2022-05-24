from . import BaseAgent
from common.registry import Registry
import gym
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
import random
import numpy as np


@Registry.register_model('rl')
class RLAgent(BaseAgent):
    def __init__(self, world, intersection_id='intersection_1_1'):
        super().__init__(world)
        self.id = intersection_id
        self.inter_obj = self.world.id2intersection[self.id]
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj,
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)
    def get_ob(self):
        return self.ob_generator.generate()

    def get_phase(self):
        return self.phase_generator.generate()

    def get_reward(self):
        reward = self.reward_generator.generate()
        assert len(reward) == 1
        return reward[0]

    def get_action(self):
        return self.action_space.sample()
    
    def sample(self):
        return random.randint(0,self.action_space.n-1)

    def get_queue(self):
        """
        get queue of intersection
        return: value
        """
        queue = []
        queue.append(self.queue.generate())
        # sum of lane nums
        queue = np.sum(np.squeeze(np.array(queue)))
        return queue

    def get_delay(self):
        """
        get delay of intersection
        return: value
        """
        delay = []
        delay.append(self.delay.generate())
        delay = np.sum(np.squeeze(np.array(delay)))
        return delay
    
    """
    def choose(self, **kwargs):
        raise NotImplementedError
    """

class State(object):
    # todo: implement as abstract class

    D_QUEUE_LENGTH = (8,)
    D_NUM_OF_VEHICLES = (8,)
    D_WAITING_TIME = (8,)
    D_MAP_FEATURE = (150,150,1,)
    D_CUR_PHASE = (1,)
    D_NEXT_PHASE = (1,)
    D_TIME_THIS_PHASE = (1,)
    D_IF_TERMINAL = (1,)
    D_HISTORICAL_TRAFFIC = (6,)

