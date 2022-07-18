from . import RLAgent
from common.registry import Registry

import numpy as np

import torch
import torch.nn as nn
from torch import optim

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
from generator.intersection_vehicle import IntersectionVehicleGenerator
from agent import utils

import gym
import os
from collections import deque

from pfrl.nn import Branched
import pfrl.initializers
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead


def lecun_init(layer, gain=1.0):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


@Registry.register_model('ppo_pfrl')
class IPPO_pfrl(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.world = world
        self.sub_agents = 1
        self.rank = rank
        self.device = torch.device('cpu')
        self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        self.replay_buffer = self.replay_buffer = deque(maxlen=self.buffer_size)

        self.phase = Registry.mapping['world_mapping']['traffic_setting'].param['phase']
        self.one_hot = Registry.mapping['world_mapping']['traffic_setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['model_setting'].param

        # get generator for each DQNAgent
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(world, self.inter, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

        self.learning_rate = Registry.mapping['model_mapping']['model_setting'].param['learning_rate']

        if self.phase:
            if self.one_hot:
                self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)
            else:
                self.ob_length = self.ob_generator.ob_length + 1
        else:
            self.ob_length = self.ob_generator.ob_length
        # set parameters of model here
        self.agent = None
        self.optimizer = None
        self._build_model()

    def __repr__():
        return self.model

    def reset(self):
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator_lane = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'],
                                                      in_only=True, average=None)
        self.ob_generator_wait = LaneVehicleGenerator(self.world, inter_obj, ['lane_waiting_count'],
                                                      in_only=True, average=None)
        self.ob_generator_wait_time = LaneVehicleGenerator(self.world, inter_obj, ['lane_waiting_time_count'],
                                                           in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_time_count"],
                                                     in_only=True, average='all', negative=True)
        self.vehicles_generator = IntersectionVehicleGenerator(self.world, inter_obj, ["lane_vehicles"])
    def get_ob(self):
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        # setting is borrowed from sumo_rl
        rewards = []
        rewards.append(self.reward_generator.generate())
        norm_rewards = [np.clip(r/224, -4, 4) for r in rewards]
        rewards = np.squeeze(np.array(norm_rewards))
        return rewards

    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        if self.phase:
            if self.one_hot:
                obs = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            else:
                obs = np.concatenate([ob, phase], axis=1)
        else:
            obs = ob
        obs = torch.tensor(obs, dtype=torch.float32)
        action = self.agent.act(obs)
        return action

    def do_observe(self, ob, phase, reward, done):
        if self.phase:
            if self.one_hot:
                obs = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            else:
                obs = np.concatenate([ob, phase], axis=1)
        else:
            obs = ob
        obs = torch.tensor(obs, dtype=torch.float32)
        self.agent.observe(obs, reward, done, False)

    def train(self):
        result = self.agent.get_statistics()
        result = result[3][1]
        return result

    def update_target_network(self):
        pass

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        # reformat it later
        self.replay_buffer.append((key, (obs, cur_phase, rewards, dones)))
        self.do_observe(ob, phase, reward, done)

    def _build_model(self):
        """
        self.agent = PPO(self.model, self.optimizer, gpu=self.device.index,
                         phi=lambda x: np.asarray(x, dtype=np.float32),
                         clip_eps=0.1,
                         clip_eps_vf=None,
                         update_interval=1024,
                         minibatch_size=256,
                         epochs=4,
                         standardize_advantages=True,
                         entropy_coef=0.001,
                         max_grad_norm=0.5)
        """
        # self.model = PPONet(self.ob_length, self.action_space.n)
        self.model = nn.Sequential(
            # lecun_init(nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2))),
            # nn.ReLU(),
            # nn.Flatten(),
            # lecun_init(nn.Linear(h*w*64, 64)),
            nn.Flatten(start_dim=1, end_dim=-1),
            lecun_init(nn.Linear(self.ob_length, 64)),
            nn.ReLU(),
            lecun_init(nn.Linear(64, 64)),
            nn.ReLU(),
            Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(64, self.action_space.n)),
                    SoftmaxCategoricalHead()
                ),
                lecun_init(nn.Linear(64, 1))
            )
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        self.agent = PPO(self.model, self.optimizer,
                         phi=lambda x: np.asarray(x, dtype=np.float32),
                         clip_eps=0.1,
                         clip_eps_vf=0.2,
                         update_interval=360,
                         minibatch_size=360,
                         epochs=4,
                         standardize_advantages=True,
                         entropy_coef=0.001,
                         max_grad_norm=0.5)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self._build_model()
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.model.state_dict(), model_name)


