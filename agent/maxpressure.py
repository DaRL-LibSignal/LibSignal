from . import BaseAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
import numpy as np
import gym

@Registry.register_model('maxpressure')
class MaxPressureAgent(BaseAgent):
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
        
        # the minimum duration of time of one phase
        self.t_min = Registry.mapping['model_mapping']['model_setting'].param['t_min']
        # self.t_min = self.inter_obj.phases_time

    def reset(self):
        # get generator for each MaxPressure
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
        # get lane pressure
        lvc = self.world.get_info("lane_count")

        # if self.inter_obj.current_phase_time < self.t_min[self.inter_obj.current_phase]:
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase

        max_pressure = None
        action = -1
        for phase_id in range(len(self.inter_obj.phases)):
            pressure = sum([lvc[start] - lvc[end] for start, end in self.inter_obj.phase_available_lanelinks[phase_id]])
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure

        return action

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