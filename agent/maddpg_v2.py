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


@Registry.register_model('maddpg_v2')
class MADDPGAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

        self.world = world
        self.sub_agents = len(self.world.id2intersection)
        self.agents = []
        # For safety, allocate sub_agents its corresponding intersection from world
        for inter_id in self.world.id2intersection:  # checked it's sorted by world
            self.agents.append(MADDPG_SUBAgent(world, inter_id))
        self.actions_list = []
        self.obs_list = []
        self.prob = []
        for ag in self.agents:
            self.actions_list.append(ag.action_space.n)
            self.obs_list.append(ag.ob_length)

        obs_dim = sum(self.obs_list)
        actions_dim = sum(self.actions_list)
        for ag in self.agents:
            ag.create_model(obs_dim, actions_dim)

    def reset(self):
        for ag in self.agents:
            ag.reset()

    def get_ob(self):
        x_obs = []
        for ag in self.agents:
            x_obs.append(ag.ob_generator.generate())
        return x_obs

    def get_reward(self):
        rewards = []
        for ag in self.agents:
            rewards.append(ag.reward_generator.generate())
        return rewards

    def get_phase(self):
        phase = []
        for ag in self.agents:
            phase.append(ag.phase_generator.generate())
        return phase
    """
    def get_action(self, obs, phase, test=False):
        self.prob = []
        actions = []
        for idx, ag in enumerate(self.agents):
            single_action_prob = ag.choose_action(obs[idx], test)
            p = single_action_prob/sum(single_action_prob)
            if test:
                action = np.argmax(p)
            else:
                action = np.random.choice(range(ag.action_space.n), p=p)
            actions.append(action)
            self.prob.append(p)
        return actions
    """
    def get_action(self, obs, phase, test=False):
        self.prob = []
        actions = []
        for idx, ag in enumerate(self.agents):
            p = ag.choose_action(obs[idx], test)

            action = np.argmax(p)
            actions.append(action)
            self.prob.append(p)
        return actions

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def get_action_prob(self, obs, phase):
        self.get_action(obs, phase)
        return np.stack(self.prob)

    def save_model(self, e=""):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_model(self, e=""):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions_prob, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        # TODO add phase and onehot later
        # last ob should be a list of observations with not necessary the same dimensional obs
        state_list = [[] for _ in self.agents]
        action_list = [[] for _ in self.agents]
        next_state_list = [[] for _ in self.agents]
        reward_list = [[] for _ in self.agents]
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        for item in samples:
            [state_list[i].append(val) for i, val in enumerate(item[1][0])]
            [action_list[i].append(val) for i, val in enumerate(item[1][2])]
            [next_state_list[i].append(val) for i, val in enumerate(item[1][4])]
            [reward_list[i].append(val) for i, val in enumerate(item[1][3])]
        for same_state, same_action, same_n_state, same_reward\
                in zip(state_list, action_list, next_state_list, reward_list):
            state_batch.append(torch.tensor(np.stack(same_state), dtype=torch.float32))
            action_batch.append(torch.tensor(np.stack(same_action), dtype=torch.float32))
            next_state_batch.append(torch.tensor(np.stack(same_n_state), dtype=torch.float32))
            reward_batch.append(torch.tensor(np.concatenate(same_reward), dtype=torch.float32))
        return state_batch, action_batch, next_state_batch, reward_batch

    def train(self):
        critic_loss_list = []
        sample_index = random.sample(range(len(self.replay_buffer)), self.batch_size)
        sample = [self.replay_buffer[idx] for idx in sample_index]
        actor_states, actions, actor_new_states, rewards = self._batchwise(sample)
        states = torch.cat(actor_states, dim=1)
        states_ = torch.cat(actor_new_states, dim=1)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        for idx, ag in enumerate(self.agents):
            #ag.actor.optimizer.zero_grad()

            new_states = actor_new_states[idx]
            new_pi = ag.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            mu_states = actor_states[idx]
            pi = ag.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[idx])
        new_actions = torch.cat(all_agents_new_actions, dim=1)
        mu = torch.cat(all_agents_new_mu_actions, dim=1)
        old_actions = torch.cat(old_agents_actions, dim=1)

        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.zero_grad()

        for idx, ag in enumerate(self.agents):
            critic_value_ = ag.target_critic.forward(states_, new_actions).flatten()
            critic_value = ag.critic.forward(states, old_actions).flatten()
            target = rewards[idx].flatten() + ag.gamma * critic_value_
            critic_loss = ag.loss(target, critic_value)

            ag.critic.optimizer.zero_grad()
            #critic_loss.backward()
            critic_loss.backward(retain_graph=True)
            clip_grad_norm_(ag.critic.parameters(), ag.grad_clip)
            ag.critic.optimizer.step()

            # TODO: test zero grad q here
            #ag.critic.optimizer.zero_grad()

            actor_loss = ag.critic.forward(states, mu).flatten()
            actor_loss = torch.mean(torch.mul(-1, actor_loss))
            #ag.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss_list.append(critic_loss.detach().cpu().numpy())

        for agent_idx, agent in enumerate(self.agents):
            clip_grad_norm_(agent.actor.parameters(), agent.grad_clip)
            agent.actor.optimizer.step()
            agent.update_network_parameters()

        return critic_loss_list

    def update_target_network(self):
        # update while training
        pass


class MADDPG_SUBAgent(object):
    def __init__(self, world, inter_id):
        self.world = world

        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']

        self.inter = inter_id
        self.inter_obj = self.world.id2intersection[inter_id]

        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))
        if self.phase:
            if self.one_hot:
                self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)
            else:
                self.ob_length = self.ob_generator.ob_length + 1
        else:
            self.ob_length = self.ob_generator.ob_length
        self.loss = nn.MSELoss(reduction='mean')

        self.actor = None
        self.target_actor = None
        self.critic = None
        self.target_critic = None

        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.tau = Registry.mapping['model_mapping']['setting'].param['tau']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.alpha = Registry.mapping['model_mapping']['setting'].param['alpha']
        self.beta = Registry.mapping['model_mapping']['setting'].param['beta']
        self.fc1 = Registry.mapping['model_mapping']['setting'].param['fc1']
        self.fc2 = Registry.mapping['model_mapping']['setting'].param['fc2']
        self.chkpt_dir = Registry.mapping['logger_mapping']['path'].path

    def create_model(self, obs_dim, actions_dim):
        # use global information from all agents to create critic
        self.actor = ActorNetwork(self.alpha, self.ob_length, self.fc1, self.fc2,
                                  self.action_space.n, self.inter + '_actor', self.chkpt_dir)
        self.critic = CriticNetwork(self.beta, obs_dim, self.fc1, self.fc2, actions_dim,
                                    self.inter + '_critic', self.chkpt_dir)
        self.target_actor = ActorNetwork(self.alpha, self.ob_length, self.fc1, self.fc2,
                                         self.action_space.n, self.inter + '_target_actor', self.chkpt_dir)
        self.target_critic = CriticNetwork(self.beta, obs_dim, self.fc1, self.fc2, actions_dim,
                                           self.inter + '_target_critic', self.chkpt_dir)

        self.update_network_parameters(tau=1)

    def reset(self):
        inter_obj = self.world.id2intersection[self.inter]
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)

    """
    def choose_action(self, observation, test=False):
        state = torch.tensor(observation[np.newaxis], dtype=torch.float32)
        actions = self.actor.forward(state)
        if test:
            return actions.detach().cpu().numpy()[0]
        noise = torch.rand(self.action_space.n)
        actions = (actions + noise).detach().cpu().numpy()[0]
        return actions
    """
    def choose_action(self, observation, test=False):
        state = torch.tensor(observation[np.newaxis], dtype=torch.float32)
        actions = self.actor.forward(state)
        actions = actions.detach().cpu().numpy()[0]
        return actions

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.pi(x)
        #pi = torch.softmax(self.pi(x), dim=1)
        pi = F.gumbel_softmax(x, dim=1)
        return pi

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, full_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims + full_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))
