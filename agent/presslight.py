from . import RLAgent
from common.registry import Registry
from agent import utils
import numpy as np
import os
import random
from collections import deque
import gym
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator


@Registry.register_model('presslight')
class PressLightAgent(RLAgent):
    '''
    PressLightAgent coordinates traffic signals by learning Max Pressure.
    '''
    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.dic_agent_conf = Registry.mapping['model_mapping']['setting']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['setting']
    
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)

    def __repr__(self):
        return self.model.__repr__()
    
    def _create_generators(self):
        super()._create_generators()
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["pressure"], average="all", negative=True)


    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
        self._create_generators()

    def get_ob(self):
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        x_obs = np.array([ob_g.generate() for ob_g in self.ob_generator], dtype=np.float32).reshape(1,-1)
        return x_obs
    
    def get_reward(self):
        '''
        get_reward
        Get reward from environment. The reward is pressure of intersections.

        :param: None
        :return rewards: rewards generated by reward_generator
        '''
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards))
        assert rewards.size == 1
        return rewards

    def get_phase(self):
        '''
        get_phase
        Get current phase of intersection(s) from environment.

        :param: None
        :return phase: current phase generated by phase_generator
        '''
        phase = [self.phase_generator.generate()]
        phase = np.array(phase, dtype=np.int8)
        return phase
    
    def get_action(self, ob, phase, test=False):
        '''
        get_action
        Generate action.

        :param ob: observation
        :param phase: current phase
        :param test: boolean, decide whether is test process
        :return: action that has the highest score
        '''
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
        actions = self.model(observation, train=False)
        actions = actions.clone().detach().numpy()
        return np.argmax(actions, axis=1)

    def sample(self):
        '''
        sample
        Sample action randomly.

        :param: None
        :return: action generated randomly.
        '''
        return np.random.randint(0, self.action_space.n, self.sub_agents)
    
    def _build_model(self):
        '''
        _build_model
        Build a DQN model.

        :param: None
        :return model: DQN model
        '''
        model = DQNNet(self.ob_length, self.action_space.n, self.dic_agent_conf)
        return model

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        '''
        remember
        Put current step information into replay buffer for training agent later.

        :param last_obs: last step observation
        :param last_phase: last step phase
        :param actions: actions executed by intersections
        :param actions_prob: the probability that the intersections execute the actions
        :param rewards: current step rewards
        :param obs: current step observation
        :param cur_phase: current step phase
        :param done: boolean, decide whether the process is done
        :param key: key to store this record, e.g., episode_step_agentid
        :return: None
        '''
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        '''
        _batchwise
        Reconstruct the samples into batch form(last state, current state, reward, action).

        :param samples: original samples record in replay buffer
        :return state_t, state_tp, rewards, actions: information with batch form
        '''
        # (64,24)
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
        # (batch_size,32)
        state_t = torch.tensor(feature_t, dtype=torch.float32)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32)
        # rewards:(64)
        rewards = torch.tensor(np.array([item[1][3] for item in samples]), dtype=torch.float32)  # TODO: BETTER WA
        # actions:(64,1)
        actions = torch.tensor(np.array([item[1][2] for item in samples]), dtype=torch.long)
        return state_t, state_tp, rewards, actions

    def train(self):
        '''
        train
        Train the agent, optimize the action generated by agent.

        :param: None
        :return: value of loss
        '''
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
        '''
        update_target_network
        Update params of target network.

        :param: None
        :return: None
        '''
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        '''
        load_model
        Load model params of an episode.

        :param e: specified episode
        :return: None
        '''
        model_name = os.path.join(
            Registry.mapping['logger_mapping']['path'].path, 'model', f'{e}_{self.rank}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))
    
    def save_model(self, e):
        '''
        save_model
        Save model params of an episode.

        :param e: specified episode, used for file name
        :return: None
        '''
        path = os.path.join(
            Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.model.state_dict(), model_name)

class DQNNet(nn.Module):
    '''
    DQNNet consists of 3 dense layers.
    '''
    def __init__(self, input_dim, output_dim, dic_agent_conf):
        super(DQNNet, self).__init__()
        self.dic_agent_conf = dic_agent_conf
        self.dense_1 = nn.Linear(input_dim, self.dic_agent_conf.param['d_dense'])
        self.dense_2 = nn.Linear(self.dic_agent_conf.param['d_dense'], self.dic_agent_conf.param['d_dense'])
        self.dense_3 = nn.Linear(self.dic_agent_conf.param['d_dense'], output_dim)

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

