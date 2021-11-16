import gym
import torch

from .import RLAgent
import numpy as np
from collections import deque
import os
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random


class DQN(nn.Module):
    def __init__(self, size_in, size_out):
        super(DQN, self).__init__()
        self.dense_1 = nn.Linear(size_in, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, size_out)

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


class DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid
        self.ob_generator = ob_generator
        self.list_feature_name = ['num_of_vehicles', 'cur_phase']
        ob_length = [State.dims["D_" + feature_name.upper()][0] for feature_name in self.list_feature_name]
        self.ob_length = sum(ob_length)

        self.memory = deque(maxlen=4000)
        self.learning_start = 4000
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.meta_test_start = 100
        self.meta_test_update_model_freq = 10
        self.meta_test_update_target_model_freq = 200
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 32

        self.criterion = nn.MSELoss()
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, alpha=0.9, centered=False, eps=1e-7)
        self.update_target_network()

    def get_ob(self):

        obs_lane = [self.ob_generator[0].generate()]
        cur_phase = [self.ob_generator[1].generate()]

        # print(obs_lane)
        state = State(
            num_of_vehicles=np.reshape(np.array(obs_lane[0]), newshape=(-1, 12)),
            cur_phase=np.reshape(np.array([cur_phase]), newshape=(-1, 1))
        )
        return state.to_list()

    def convert_state_to_input(self, state):
        ''' convert a state struct to the format for neural network input'''
        return [getattr(state, feature_name)
                for feature_name in self.list_feature_name]

    def choose(self, ob):
        # ob = self.convert_state_to_input(state)
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        ob = torch.tensor(np.concatenate((ob[0], ob[1]), axis=1)).float()
        act_values = self.model.forward(ob)
        return torch.argmax(act_values, dim=1)

    def get_action(self, ob):
        ob = torch.tensor(np.concatenate((ob[0], ob[1]), axis=1)).float()
        act_values = self.model.forward(ob)
        return torch.argmax(act_values, dim=1)

    def sample(self):
        return self.action_space.sample()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = DQN(self.ob_length, self.action_space.n)
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))


    def _encode_sample(self, minibatch):
        obses_t, actions_t, rewards_t, obses_tp1 = list(zip(*minibatch))
        obs = [np.squeeze(obs_i) for obs_i in list(zip(*obses_t))]
        # add generality after finish presslight
        obs = np.concatenate((obs[0], obs[1][:, np.newaxis]), axis=1)
        next_obs = [np.squeeze(obs_i) for obs_i in list(zip(*obses_tp1))]
        next_obs = np.concatenate((next_obs[0], next_obs[1][:, np.newaxis]), axis=1)
        #actions = np.array(actions_t, copy=False)
        rewards = np.array(rewards_t, copy=False)
        obs = torch.from_numpy(obs).float()
        #actions = torch.from_numpy(actions).int()
        rewards = torch.from_numpy(rewards).float()
        next_obs = torch.from_numpy(next_obs).float()
        return obs, actions_t, rewards, next_obs

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs = self._encode_sample(minibatch)
        out = self.target_model.forward(next_obs, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model.forward(obs, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model.forward(obs, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", e = 0):
        name = "dqn_agent_{}_{}.pt".format(self.iid, e)
        model_name = os.path.join(dir, name)
        self.model = DQN(self.ob_length, self.action_space.n)
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self, dir="model/dqn", e = 0):
        name = "dqn_agent_{}_{}.pt".format(self.iid, e)
        model_name = os.path.join(dir, name)
        torch.save(self.model.state_dict(), model_name)

class State(object):
    # ==========================
    dims = {
        "D_NUM_OF_VEHICLES":  (12,),
        "D_CUR_PHASE":  (1,)

    }

    # ==========================

    def __init__(self, num_of_vehicles, cur_phase):

        self.num_of_vehicles = num_of_vehicles
        self.cur_phase = cur_phase

    def to_list(self):
        results = [self.num_of_vehicles,self.cur_phase]
        return results