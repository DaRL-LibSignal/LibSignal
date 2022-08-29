from . import BaseAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
import numpy as np
import gym

@Registry.register_model('fixedtime')
class FixedTimeAgent(BaseAgent):
    """
    Agent using Max-Pressure method to control traffic light
    """
    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        self.model = None

        # get generator for each MaxPressure
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                     in_only=True, average='all', negative=True)
        
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)

        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True,
                                                     negative=False) 

        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        # dirrerent datasets have the same t_fixed
        self.t_fixed = Registry.mapping['model_mapping']['model_setting'].param['t_fixed']
        
    def reset(self):
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                     in_only=True, average='all', negative=True)
        
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)

        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True,
                                                     negative=False)

    def get_ob(self):
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards
    
    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        # phase = np.concatenate(phase, dtype=np.int8)
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase
    
    def get_action(self, ob, phase, test=True):
        # phases: just index of self.inter_obj.phases, not green light index
        assert self.inter_obj.current_phase == phase[-1]
        if self.inter_obj.current_phase_time < self.t_fixed:
            return self.inter_obj.current_phase
        else:
            return (self.inter_obj.current_phase+1) % len(self.inter_obj.phases)

    def get_queue(self):
        queue = []
        queue.append(self.queue.generate())
        queue = np.sum(np.squeeze(np.array(queue)))
        return queue

    def get_delay(self):
        delay = []
        delay.append(self.delay.generate())
        delay = np.sum(np.squeeze(np.array(delay)))
        return delay