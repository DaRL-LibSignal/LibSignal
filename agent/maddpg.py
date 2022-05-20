from . import RLAgent

from common.registry import Registry
from collections import deque
import random
import os

from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from agent import utils

import gym
import numpy as np

from torch import nn
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


@Registry.register_model('maddpg')
class MADDPGAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1
        self.rank = rank
        self.n_intersections = len(world.id2intersection)
        self.agents = None

        self.phase = Registry.mapping['world_mapping']['traffic_setting'].param['phase']
        self.one_hot = Registry.mapping['world_mapping']['traffic_setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['model_setting'].param

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

        self.gamma = Registry.mapping['model_mapping']['model_setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['model_setting'].param['grad_clip']
        self.epsilon_decay = Registry.mapping['model_mapping']['model_setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['model_setting'].param['epsilon_min']
        self.epsilon = Registry.mapping['model_mapping']['model_setting'].param['epsilon']
        self.learning_rate = Registry.mapping['model_mapping']['model_setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['model_setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['model_setting'].param['batch_size']

        self.best_epoch = 0
        # param
        self.local_q_learn = Registry.mapping['model_mapping']['model_setting'].param['local_q_learn']
        self.last_action = 0
        self.q_length = 0

        # net works
        self.q_model = None
        self.target_q_model = None
        self.p_model = None
        self.target_p_model = None

        self.criterion = None
        self.q_optimizer = None
        self.p_optimizer = None

    def link_agents(self, agents):
        self.agents = agents
        if not self.local_q_learn:
            full_action = 0
            full_observe = 0
            for ag in self.agents:
                full_action += ag.action_space.n
                full_observe += ag.ob_length
            self.q_length = full_observe + full_action
        else:
            self.q_length = self.ob_length + self.action_space.n

        self.q_model = self._build_model(self.q_length, 1)
        self.target_q_model = self._build_model(self.q_length, 1)
        self.p_model = self._build_model(self.ob_length, self.action_space.n)
        self.target_p_model = self._build_model(self.ob_length, self.action_space.n)
        self.update_target_network()

        self.criterion = nn.MSELoss(reduction='mean')
        self.q_optimizer = optim.Adam(self.q_model.parameters(), lr=self.learning_rate)
        self.p_optimizer = optim.Adam(self.p_model.parameters(), lr=self.learning_rate)

    def reset(self):
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)

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
        phase = np.concatenate(phase, dtype=np.int8)
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
        actions = self.p_model(observation, train=True)
        actions = self.G_softmax(actions)
        actions = actions.clone().detach().numpy()
        actions = np.argmax(actions, axis=1)
        return actions

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def G_softmax(self, p):
        """
        u = torch.rand(self.action_space.n)
        prob = F.softmax(p - torch.log(-torch.log(u)), dim=1)
        """
        prob = F.gumbel_softmax(p)
        return prob

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
        rewards = torch.tensor(np.array([item[1][3] for item in samples])[:, np.newaxis], dtype=torch.float32)  # TODO: BETTER WA
        # TODO: reshape
        actions = torch.tensor(np.array([utils.idx2onehot(item[1][2], self.action_space.n).squeeze() for item in samples]),
                               dtype=torch.long)
        return state_t, state_tp, rewards, actions

    def train(self):
        b_t_list = []
        b_tp_list = []
        rewards_list = []
        action_list = []
        target_q = 0.0
        sample_index = random.sample(list(range(self.buffer_size)), self.batch_size)
        for ag in self.agents:
            samples = np.array(list(ag.replay_buffer), dtype=object)[sample_index]
            b_t, b_tp, rewards, actions = self._batchwise(samples)
            b_t_list.append(b_t)
            b_tp_list.append(b_tp)
            rewards_list.append(rewards)
            action_list.append(actions)
        target_act_next_list = []
        for i, ag in enumerate(self.agents):
            target_act_next = self.target_p_model(b_tp_list[i], train=False)
            target_act_next = self.G_softmax(target_act_next)
            target_act_next_list.append(target_act_next)
        full_b_t = torch.cat(b_t_list, dim=1)
        full_b_tp = torch.cat(b_tp_list, dim=1)
        full_action_tp = torch.cat(target_act_next_list, dim=1)
        full_action_t = torch.cat(action_list, dim=1)
        # combine b_t and corresponding full_actions
        if self.local_q_learn:
            q_input_target = torch.cat((b_tp_list[self.rank], action_list[self.rank]), dim=1)
            q_input = torch.cat((b_t_list[self.rank], full_action_t), dim=1)
        else:
            q_input_target = torch.cat((full_b_tp, full_action_tp), dim=1)
            q_input = torch.cat((full_b_t, full_action_t), dim=1)

        target_q_next = self.target_q_model(q_input_target, train=False)
        target_q += rewards_list[self.rank] + self.gamma * target_q_next

        q = self.q_model(q_input, train=True)

        # update q network
        q_reg = torch.mean(torch.square(q))
        q_loss = self.criterion(q, target_q)
        loss_of_q = q_loss + 1e-3 * q_reg

        # update p network
        p = self.p_model(b_t_list[self.rank], train=True)
        p_reg = torch.mean(torch.square(p))

        p_prob = self.G_softmax(p)
        if self.local_q_learn:
            pq_input = torch.cat((b_t_list[self.rank], action_list[self.rank]), dim=1)
        else:
            action_list[self.rank] = p_prob
            full_action_t_q = torch.cat(action_list, dim=1)
            pq_input = torch.cat((full_b_t, full_action_t_q), dim=1)

        p_loss = -torch.mean(self.q_model(pq_input, train=False))
        loss_of_p = p_loss + p_reg * 1e-3

        self.q_optimizer.zero_grad()
        loss_of_q.backward()
        clip_grad_norm_(self.q_model.parameters(), self.grad_clip)
        self.q_optimizer.step()

        self.p_optimizer.zero_grad()
        loss_of_p.backward()
        clip_grad_norm_(self.p_model.parameters(), self.grad_clip)
        self.p_optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # TODO: q loss or p loss ?
        return loss_of_q.clone().detach().numpy()

    def remember(self, last_obs, last_phase, actions, rewards, obs, cur_phase, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _build_model(self, input_dim, output_dim):
        model = DQNNet(input_dim, output_dim)
        return model

    def update_target_network(self):
        q_weights = self.q_model.state_dict()
        self.target_q_model.load_state_dict(q_weights)
        p_weights = self.p_model.state_dict()
        self.target_p_model.load_state_dict(p_weights)

    def load_model(self, e):
        model_p_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                    'model_p', f'{e}_{self.rank}.pt')
        model_q_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                    'model_q', f'{e}_{self.rank}.pt')
        self.model_q = self._build_model(self.q_length, 1)
        self.model_p = self._build_model(self.ob_length, self.action_space.n)
        self.model_q.load_state_dict(torch.load(model_q_name))
        self.model_p.load_state_dict(torch.load(model_p_name))
        self.update_target_network()

    def save_model(self, e):
        path_p = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model_p')
        path_q = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model_q')
        if not os.path.exists(path_p):
            os.makedirs(path_p)
        if not os.path.exists(path_q):
            os.makedirs(path_q)
        model_p_name = os.path.join(path_p, f'{e}_{self.rank}.pt')
        model_q_name = os.path.join(path_q, f'{e}_{self.rank}.pt')
        torch.save(self.p_model.state_dict(), model_p_name)
        torch.save(self.q_model.state_dict(), model_q_name)

    def load_best_model(self,):
        model_p_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                    'model_p', f'{self.best_epoch}_{self.rank}.pt')
        model_q_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                    'model_q', f'{self.best_epoch}_{self.rank}.pt')
        self.q_model.load_state_dict(torch.load(model_q_name))
        self.p_model.load_state_dict(torch.load(model_p_name))
        self.update_target_network()


class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, output_dim)

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
