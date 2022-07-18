from . import RLAgent
from common.registry import Registry
from agent import utils
import numpy as np
import os
import random
from collections import deque
import gym

from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


@Registry.register_model('dqn')
class DQNAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1
        self.rank = rank

        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']

        # get generator for each DQNAgent
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world,  self.inter, ['lane_count'], in_only=True, average=None)

        self.phase_generator = IntersectionPhaseGenerator(world,  self.inter, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world,  self.inter, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

        if self.phase:
            if self.one_hot:
                self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)
            else:
                self.ob_length = self.ob_generator.ob_length + 1
        else:
            self.ob_length = self.ob_generator.ob_length

        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)

    def __repr__(self):
        return self.model.__repr__()

    def reset(self):
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.queue = LaneVehicleGenerator(self.world, inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
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

    def get_action(self, ob, phase, test=False):
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        if self.phase:
            if self.one_hot:
                feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            else:
                feature = np.concatenate([ob, phase], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        # TODO: no need to calculate gradient when interacting with environment
        actions = self.model(observation, train=False)
        actions = actions.clone().detach().numpy()
        return np.argmax(actions, axis=1)

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def _build_model(self):
        model = DQNNet(self.ob_length, self.action_space.n)
        return model

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        obs_t = np.concatenate([item[1][0] for item in samples])
        obs_tp = np.concatenate([item[1][4] for item in samples])
        if self.phase:
            if self.one_hot:
                phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n) for item in samples])
                phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n) for item in samples])
            else:
                phase_t = np.concatenate([item[1][1] for item in samples])
                phase_tp = np.concatenate([item[1][5] for item in samples])
            feature_t = np.concatenate([obs_t, phase_t], axis=1)
            feature_tp = np.concatenate([obs_tp, phase_tp], axis=1)
        else:
            feature_t = obs_t
            feature_tp = obs_tp
        state_t = torch.tensor(feature_t, dtype=torch.float32)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32)
        rewards = torch.tensor(np.array([item[1][3] for item in samples]), dtype=torch.float32)  # TODO: BETTER WA
        actions = torch.tensor(np.array([item[1][2] for item in samples]), dtype=torch.long)
        return state_t, state_tp, rewards, actions

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        out = self.target_model(b_tp, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(b_t, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model(b_t, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, output_dim)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)
