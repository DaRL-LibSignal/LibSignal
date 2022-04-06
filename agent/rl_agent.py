from . import BaseAgent
from common.registry import Registry
import gym
from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator


@Registry.register_model('rl')
class RLAgent(BaseAgent):
    def __init__(self, world, intersection_id='intersection_1_1'):
        super().__init__()
        self.id = intersection_id
        self.world = world
        self.action_space = gym.spaces.Discrete(len(world.id2intersection[intersection_id].phases))
        self.ob_generator = LaneVehicleGenerator(self.world, world.id2intersection[self.id],
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, world.id2intersection[self.id],
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, world.id2intersection[self.id],
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)

    def get_ob(self):
        return self.ob_generator.generate()

    def get_phase(self):
        return self.phase_generator.generate()

    def get_reward(self):
        reward = self.reward_generator.generate()
        assert len(reward) == 1
        return reward[0]

    def get_action(self, ob, phase):
        return self.action_space.sample()
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

