from . import BaseAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
import numpy as np
import gym


@Registry.register_model('sotl')
class SOTLAgent(BaseAgent):
    """
    Agent using Self-organizing Traffic Light(SOTL) Control method to control traffic light
    """
    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        # some threshold to deal with phase requests
        self.min_green_vehicle = Registry.mapping['model_mapping']['model_setting'].param['min_green_vehicle']
        self.max_red_vehicle = Registry.mapping['model_mapping']['model_setting'].param['max_red_vehicle']
        self.t_min = Registry.mapping['model_mapping']['model_setting'].param['t_min']
        # get generator for each SOTL
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.model = None
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_waiting_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

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
        lane_waiting_count = self.world.get_info("lane_waiting_count")
        action = self.inter.current_phase
        if self.inter.current_phase_time >= self.t_min:
            num_green_vehicles = sum([lane_waiting_count[lane] for
                                      lane in self.inter.phase_available_startlanes[self.inter.current_phase]])
            num_red_vehicles = sum([lane_waiting_count[lane] for lane in self.inter.startlanes])
            num_red_vehicles -= num_green_vehicles

            if num_green_vehicles <= self.min_green_vehicle and num_red_vehicles > self.max_red_vehicle:
                action = (action + 1) % self.action_space.n

        return action
