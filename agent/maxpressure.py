from . import BaseAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
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

        # the minimum duration of time of one phase
        self.t_min = Registry.mapping['model_mapping']['model_setting'].param['t_min']
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.model = None
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True,
                                                 average=None)
        self.phase_generator = IntersectionPhaseGenerator(world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

    def reset(self):
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)

    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = np.concatenate(phase, dtype=np.int8)
        return phase

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

    def get_action(self, ob, phase, test=True):
        lvc = self.world.get_info("lane_count")
        if self.inter.current_phase_time < self.t_min:
            return self.inter.current_phase
        max_pressure = None
        action = -1
        for phase_id in range(len(self.inter.phases)):
            # todo: implement with sumo setting.
            pressure = sum([lvc[start] - lvc[end] for start, end in self.inter.phase_available_lanelinks[phase_id]])
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure
        return [action]
