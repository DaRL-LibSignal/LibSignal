from . import RLAgent
import random
import numpy as np
from collections import deque
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.registry import Registry
import gym
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
from torch.nn.utils import clip_grad_norm_
from agent import utils


@Registry.register_model('frap')
class FRAP_DQNAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world,world.intersection_ids[rank])
        self.dic_agent_conf = Registry.mapping['model_mapping']['setting']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['setting']
        
        self.gamma = self.dic_agent_conf.param["gamma"]
        self.grad_clip = self.dic_agent_conf.param["grad_clip"]
        self.epsilon = self.dic_agent_conf.param["epsilon"]
        self.epsilon_min = self.dic_agent_conf.param["epsilon_min"]
        self.epsilon_decay = self.dic_agent_conf.param["epsilon_decay"]
        self.learning_rate = self.dic_agent_conf.param["learning_rate"]
        self.batch_size = self.dic_agent_conf.param["batch_size"]
        self.num_phases = len(self.dic_traffic_env_conf.param["phases"])
        self.num_actions = len(self.dic_traffic_env_conf.param["phases"])
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1
        self.rank = rank

        self.phase = self.dic_agent_conf.param['phase']
        self.one_hot = self.dic_agent_conf.param['one_hot']

        # get generator for each Agent
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj,
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)

        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)

        map_name = Registry.mapping['command_mapping']['setting'].param['network']
        self.phase_pairs = self.dic_traffic_env_conf.param['signal_config'][map_name]['phase_pairs']
        self.comp_mask = self.relation()
        self.dic_phase_expansion = None
        # if self.phase:
        #     if self.one_hot:
        #         if self.ob_generator.ob_length == 8:
        #             self.dic_phase_expansion = self.dic_traffic_env_conf.param["phase_expansion_8"]

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.criterion = nn.MSELoss(reduction='mean')

    def __repr__(self):
        return self.model.__repr__()

    def reset(self):
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj,
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)

    def relation(self):
        comp_mask = []
        for i in range(len(self.phase_pairs)):
            zeros = np.zeros(len(self.phase_pairs) - 1, dtype=np.int)
            cnt = 0
            for j in range(len(self.phase_pairs)):
                if i == j: continue
                pair_a = self.phase_pairs[i]
                pair_b = self.phase_pairs[j]
                if len(list(set(pair_a + pair_b))) == 3: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = torch.from_numpy(np.asarray(comp_mask))
        return comp_mask 


    def _build_model(self):
        model = FRAP(self.dic_agent_conf, self.dic_phase_expansion, self.num_actions, self.phase_pairs, self.comp_mask)
        return model

    def get_ob(self):
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs #(1,12)

    def get_reward(self):
        rewards = []
        rewards.append(self.reward_generator.generate())
        # TODO check whether to multiply 12
        rewards = np.squeeze(np.array(rewards)) * self.num_phases
        return rewards

    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        """
        ob:(1,12)
        phase:(1,)
        """
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        # 12->8 
        # ob = utils.remove_right_lane(ob)
        if self.phase:
            if self.one_hot:
                feature_p = utils.idx2onehot(phase, self.action_space.n,self.dic_phase_expansion)
                # feature_p = utils.idx2onehot(phase, self.action_space.n)
                feature = np.concatenate([feature_p, ob], axis=1)
            else:
                feature = np.concatenate([phase.reshape(1,-1), ob], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions = self.model(observation, train=True) #1, 8
        actions = actions.clone().detach().numpy()
        return np.argmax(actions, axis=1)

    def sample(self):
        return np.random.randint(0, self.num_phases, self.sub_agents)

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        # (batch_size,12)
        obs_t_all=[item[1][0] for item in samples] # last_obs(batch, 1, lane_num)
        obs_tp_all=[item[1][4] for item in samples] # cur_obs
        # obs_t = [utils.remove_right_lane(x) for x in obs_t_all]
        # obs_tp = [utils.remove_right_lane(x) for x in obs_tp_all]
        obs_t = obs_t_all
        obs_tp = obs_tp_all
        obs_t = np.concatenate(obs_t) # (batch,lane_num)
        obs_tp = np.concatenate(obs_tp) # (batch,lane_num)
        if self.phase:
            if self.one_hot:
                phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n, self.dic_phase_expansion) for item in samples])
                phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n, self.dic_phase_expansion) for item in samples])
            else:
                phase_t = np.concatenate([item[1][1].reshape(1,-1) for item in samples]) # (batch, 1)
                phase_tp = np.concatenate([item[1][5].reshape(1,-1) for item in samples])
            feature_t = np.concatenate([phase_t, obs_t], axis=1) # (batch,ob_length)
            feature_tp = np.concatenate([phase_tp, obs_tp], axis=1)
        else:
            feature_t = obs_t
            feature_tp = obs_tp
        # (batch_size, ob_length)
        state_t = torch.tensor(feature_t, dtype=torch.float32)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32)
        # rewards:(64)
        rewards = torch.tensor(np.array([item[1][3] for item in samples]), dtype=torch.float32)  # TODO: BETTER WA
        # actions:(64,1)
        actions = torch.tensor(np.array([item[1][2] for item in samples]), dtype=torch.long)
        return state_t, state_tp, rewards, actions

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        out = self.target_model(b_tp, train=False) # (batch_size,num_actions)
        target = rewards + self.gamma * torch.max(out, dim=1)[0] # (batch_size)
        target_f = self.model(b_t, train=False) # (batch_size,num_actions)
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

    def load_model(self, e):
        model_name = os.path.join(
            Registry.mapping['logger_mapping']['path'].path, 'model', f'{e}_{self.rank}.pt')
        self.model = FRAP(self.dic_agent_conf, self.dic_phase_expansion, self.num_actions, self.phase_pairs, self.comp_mask)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = FRAP(self.dic_agent_conf, self.dic_phase_expansion, self.num_actions, self.phase_pairs, self.comp_mask)
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(
            Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.model.state_dict(), model_name)


class FRAP(nn.Module):
    def __init__(self, dic_agent_conf, dic_phase_expansion, output_shape, phase_pairs, competition_mask):
        super(FRAP, self).__init__()
        self.dic_phase_expansion = dic_phase_expansion
        self.oshape = output_shape
        self.phase_pairs = phase_pairs
        self.comp_mask = competition_mask
        self.demand_shape = dic_agent_conf.param['demand_shape']      # Allows more than just queue to be used
        self.one_hot = dic_agent_conf.param['one_hot']
        self.d_out = 4      # units in demand input layer
        self.p_out = 4      # size of phase embedding
        self.lane_embed_units = 16
        relation_embed_size = 4

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)

        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)

        self.lane_conv = nn.Conv2d(2*self.lane_embed_units, 20, kernel_size=(1, 1))

        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, 20, kernel_size=(1, 1))

        self.hidden_layer = nn.Conv2d(20, 20, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(20, 1, kernel_size=(1, 1))

    def _forward(self, states):
        '''
        :params states: [agents, ob_length]
        ob_length:concat[len(one_phase),len(intersection_lane)]
        '''
        # if lane_num=12,then num_movements=12, but turning right do not be used
        num_movements = int((states.size()[1]-1)/self.demand_shape) if not self.one_hot else int((states.size()[1]-len(self.phase_pairs))/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, :1].to(torch.int64) if not self.one_hot else states[:, :len(self.phase_pairs)].to(torch.int64)
        states = states[:, 1:] if not self.one_hot else states[:, len(self.phase_pairs):]
        states = states.float()
        
        # Expand action index to mark demand input indices
        extended_acts = []
        if not self.one_hot:
            for i in range(batch_size):
                act_idx = acts[i]
                pair = self.phase_pairs[act_idx]
                zeros = torch.zeros(num_movements, dtype=torch.int64)
                zeros[pair[0]] = 1
                zeros[pair[1]] = 1
                extended_acts.append(zeros)
            extended_acts = torch.stack(extended_acts)
        else:
            extended_acts = acts
        phase_embeds = torch.sigmoid(self.p(extended_acts))

        phase_demands = []
        # if num_movements == 12:
        #     order_lane = [0,1,3,4,6,7,9,10] # remove turning_right phase
        # else:
        #     order_lane = [i for i in range(num_movements)]
        # for idx, i in enumerate(order_lane):
        for i in range(num_movements):
            # phase = phase_embeds[:, idx]  # size 4
            phase = phase_embeds[:, i]  # size 4
            demand = states[:, i:i+self.demand_shape]
            demand = torch.sigmoid(self.d(demand))    # size 4
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands_old = torch.stack(phase_demands, 1)
        # turn direction from NESW to ESWN
        if num_movements == 8:
            phase_demands = torch.cat([phase_demands_old[:,2:,:],phase_demands_old[:,:2,:]],1)
        elif num_movements == 12:
            phase_demands = torch.cat([phase_demands_old[:,3:,:],phase_demands_old[:,:3,:]],1)
        # phase_demands = torch.stack(phase_demands, 1)


        pairs = []
        for pair in self.phase_pairs:
            pairs.append(phase_demands[:, pair[0]] + phase_demands[:, pair[1]])

        rotated_phases = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i != j: rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        rotated_phases = torch.stack(rotated_phases, 1)
        rotated_phases = torch.reshape(rotated_phases,
                                       (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units))
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        rotated_phases = F.relu(self.lane_conv(rotated_phases))  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1))
        relations = F.relu(self.relation_embedding(competition_mask))
        relations = relations.permute(0, 3, 1, 2)  # Move channels up
        relations = F.relu(self.relation_conv(relations))  # Pair demand representation

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # Pairwise competition result

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1))
        q_values = (lambda x: torch.sum(x, dim=2))(combine_features) # (b,8)
        return q_values
        

    def forward(self, states, train=True):
        if train:
            return self._forward(states)
        else:
            with torch.no_grad():
                return self._forward(states)
