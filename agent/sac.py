from . import RLAgent
from common.registry import Registry
from agent import utils
import numpy as np
import os
import random
from collections import deque
import gym
import copy
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.distributions as distributions

@Registry.register_model('sac')
class SACAgent(RLAgent):
    '''
    SACAgent determines each intersection's action with its own intersection information.
    '''
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
                self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)  ##=16
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
        self.temperature = Registry.mapping['model_mapping']['setting'].param['sac_temperature']

        self.actor_net, self.v_evaluate_net, self.v_target_net, self.q0_net, self.q1_net = self._build_model()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=3e-4)
        self.v_optimizer = optim.Adam(self.v_evaluate_net.parameters(), lr=3e-4)
        self.q0_optimizer = optim.Adam(self.q0_net.parameters(), lr=3e-4)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=3e-4)

        self.target_model = self._build_model()
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        self.add_act_to_buffer = True

    def __repr__(self):
        return self.actor_net.__repr__()

    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
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
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        '''
        get_reward
        Get reward from environment.

        :param: None
        :return rewards: rewards generated by reward_generator
        '''
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards

    def get_phase(self):
        '''
        get_phase
        Get current phase of intersection(s) from environment.

        :param: None
        :return phase: current phase generated by phase_generator
        '''
        phase = []
        phase.append(self.phase_generator.generate())
        # phase = np.concatenate(phase, dtype=np.int8)
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        '''
        get_action
        Generate action.

        :param ob: observation
        :param phase: current phase
        :param test: boolean, decide whether is test process
        :return action: action that has the highest score
        '''
        if self.phase:
            if self.one_hot:
                feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            else:
                feature = np.concatenate([ob, phase], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        # TODO: no need to calculate gradient when interacting with environment
        prob_tensor = self.actor_net(observation, train=False)
        action_tensor = distributions.Categorical(prob_tensor).sample()
        action = action_tensor.numpy()
        return action

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
        Build a SAC model.

        :param: None
        :return model: SAC model
        '''
        a = self.action_space.n
        # create actor
        actor_net = SACNet(self.ob_length, self.action_space.n,output_activator=1)
        # create V critic
        v_evaluate_net = VNet(self.ob_length, 1)
        v_target_net = copy.deepcopy(v_evaluate_net)
        # create Q critic
        q0_net = SACNet(self.ob_length, self.action_space.n)
        q1_net = SACNet(self.ob_length, self.action_space.n)

        return actor_net, v_evaluate_net, v_target_net, q0_net, q1_net

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

    def update_net(self, target_net, evaluate_net, learning_rate=0.001):
        for target_param, evaluate_param in zip(
                target_net.parameters(), evaluate_net.parameters()):
            target_param.data.copy_(learning_rate * evaluate_param.data
                    + (1 - learning_rate) * target_param.data)

    def train(self):
        '''
        train
        Train the agent, optimize the action generated by agent.

        :param: None
        :return: value of loss
        '''
        samples = random.sample(self.replay_buffer, self.batch_size)
        states, next_states, rewards, actions = self._batchwise(samples)
        state_tensor = torch.as_tensor(states, dtype=torch.float)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)

        #update Q0net
        next_v_tensor = self.v_target_net(next_state_tensor)
        q_target_tensor = reward_tensor.unsqueeze(1) + self.gamma * next_v_tensor
        all_q0_pred_tensor = self.q0_net(state_tensor)
        q0_pred_tensor = torch.gather(all_q0_pred_tensor, 1, action_tensor)
        q0_loss_tensor = self.criterion(q0_pred_tensor, q_target_tensor.detach())
        self.q0_optimizer.zero_grad()
        q0_loss_tensor.backward()
        clip_grad_norm_(self.q0_net.parameters(), self.grad_clip)
        self.q0_optimizer.step()

        # update Q1net
        all_q1_pred_tensor = self.q1_net(state_tensor)
        q1_pred_tensor = torch.gather(all_q1_pred_tensor, 1, action_tensor)
        q1_loss_tensor = self.criterion(q1_pred_tensor, q_target_tensor.detach())
        self.q1_optimizer.zero_grad()
        q1_loss_tensor.backward()
        clip_grad_norm_(self.q1_net.parameters(), self.grad_clip)
        self.q1_optimizer.step()

        # update V critic
        q0_tensor = self.q0_net(state_tensor)
        q1_tensor = self.q1_net(state_tensor)
        q01_tensor = torch.min(q0_tensor, q1_tensor)
        prob_tensor = self.actor_net(state_tensor)
        ln_prob_tensor = torch.log(prob_tensor.clamp(1e-6, 1.))
        entropic_q01_tensor = prob_tensor * (q01_tensor - self.temperature * ln_prob_tensor)
        v_target_tensor = torch.sum(entropic_q01_tensor, dim=-1, keepdim=True)
        v_pred_tensor = self.v_evaluate_net(state_tensor)
        v_loss_tensor = self.criterion(v_pred_tensor, v_target_tensor.detach())
        self.v_optimizer.zero_grad()
        v_loss_tensor.backward()
        self.v_optimizer.step()
        clip_grad_norm_(self.v_evaluate_net.parameters(), self.grad_clip)
        self.update_net(self.v_target_net, self.v_evaluate_net)

        # update actor
        q0_tensor = self.q0_net(state_tensor)
        # update actor
        prob_q_tensor = prob_tensor * (self.temperature * ln_prob_tensor - q0_tensor)
        actor_loss_tensor = prob_q_tensor.sum(axis=-1).mean()
        self.update_net(self.v_target_net, self.v_evaluate_net)
        self.add_act_to_buffer = True
        self.actor_optimizer.zero_grad()
        actor_loss_tensor.backward()
        clip_grad_norm_(self.actor_net.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        # return actor_loss_tensor.clone().detach().numpy(),q0_loss_tensor.clone().detach().numpy(), \
        #     v_loss_tensor.clone().detach().numpy(), self.add_act_to_buffer
        return actor_loss_tensor.clone().detach().numpy()


    def update_target_network(self):
        '''
        update_target_network
        Update params of target network.

        :param: None
        :return: None
        '''
        weights = self.v_evaluate_net.state_dict()
        self.v_target_net.load_state_dict(weights)

    def load_model(self, e):
        '''
        load_model
        Load model params of an episode.

        :param e: specified episode
        :return: None
        '''
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self.actor_net.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        '''
        save_model
        Save model params of an episode.

        :param e: specified episode, used for file name
        :return: None
        '''
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.actor_net.state_dict(), model_name)

class SACNet(nn.Module):
    '''
    QNet ActorNet consists of 4 dense layers.
    '''
    def __init__(self, input_dim, output_dim,output_activator=False):
        super(SACNet, self).__init__()
        self.output_activator = output_activator

        self.dense_1 = nn.Linear(input_dim, 64)
        self.dense_2 = nn.Linear(64, 128)
        self.dense_3 = nn.Linear(128, 64)
        self.dense_4 = nn.Linear(64, output_dim)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = self.dense_4(x)
        if self.output_activator:
            x = F.softmax(x, dim=1)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)

class VNet(nn.Module):
    '''
    VNet consists of 3 dense layers.
    '''
    def __init__(self, input_dim, output_dim):
        super(VNet, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 32)
        self.dense_2 = nn.Linear(32, 64)
        self.dense_3 = nn.Linear(64, output_dim)

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