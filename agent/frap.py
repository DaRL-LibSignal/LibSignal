
from ctypes import util
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
        self.dic_agent_conf = Registry.mapping['model_mapping']['model_setting']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['traffic_setting']
        
        self.gamma = self.dic_agent_conf.param["gamma"]
        self.grad_clip = self.dic_agent_conf.param["grad_clip"]
        self.epsilon = self.dic_agent_conf.param["epsilon"]
        self.epsilon_min = self.dic_agent_conf.param["epsilon_min"]
        self.epsilon_decay = self.dic_agent_conf.param["epsilon_decay"]
        self.learning_rate = self.dic_agent_conf.param["learning_rate"]
        self.batch_size = self.dic_agent_conf.param["batch_size"]
        self.num_phases = len(self.dic_traffic_env_conf.param["phases"])
        self.num_actions = len(self.dic_traffic_env_conf.param["phases"])
        self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1
        self.rank = rank

        self.phase = self.dic_traffic_env_conf.param['phase']
        self.one_hot = self.dic_traffic_env_conf.param['one_hot']

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
        # phase:12-4=8
        if self.phase:
            if self.one_hot:
                if self.num_phases == 2:
                    self.ob_length = self.ob_generator.ob_length - 4 + 4 # 8+4=12
                    self.dic_phase_expansion = self.dic_traffic_env_conf.param["phase_expansion_4_lane"]
                else:
                    self.ob_length = self.ob_generator.ob_length - 4 + 8 # 8+8=16
                    self.dic_phase_expansion = self.dic_traffic_env_conf.param["phase_expansion"]
            else:
                self.ob_length = self.ob_generator.ob_length - 4 + 1 # 8+1=9
        else:
            self.ob_length = self.ob_generator.ob_length - 4 # 12-4=8

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.criterion = nn.MSELoss(reduction='mean')
        # self.optimizer = optim.RMSprop(self.model.parameters(),
        #                                lr=self.learning_rate,
        #                                alpha=0.9, centered=False, eps=1e-7)

        # self.action = 0
        # self.last_action = 0

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

    def _build_model(self):
        model = FRAP(
            self.dic_agent_conf, self.dic_traffic_env_conf, self.num_actions, self.ob_length, self.num_phases)
        return model

    # # TODO check whether to save
    # def convert_state_to_input(self, s):
    #     inputs = {}
    #     # get one hot dic
    #     if self.num_phases == 2:
    #         dic_phase_expansion = self.dic_traffic_env_conf.param["phase_expansion_4_lane"]
    #     else:
    #         dic_phase_expansion = self.dic_traffic_env_conf.param["phase_expansion"]
    #     for feature in self.dic_traffic_env_conf.param["list_state_feature"]:
    #         if feature == "cur_phase":
    #             # size:(1,action_space)--(1,8)
    #             inputs[feature] = np.array([dic_phase_expansion[s[feature]+1]])
    #         else:
    #             # size:(1,lane_num)--(1,12)
    #             inputs[feature] = np.array(s[feature])
    #     return inputs

    # # TODO check whether to save
    # def to_tensor(self, state):
    #     output = {}
    #     for i in state:
    #         output[i] = torch.from_numpy(state[i]).float()
    #         # output[i] = torch.tensor(state[i], dtype=torch.float32)
    #     return output

    def get_ob(self):
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs #(1,12)

    def get_reward(self):
        rewards = []
        rewards.append(self.reward_generator.generate())
        # TODO check whether to multiply 12
        rewards = np.squeeze(np.array(rewards)) * 12
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
        ob = utils.remove_right_lane(ob)
        if self.phase:
            if self.one_hot:
                feature_p = utils.idx2onehot(phase, self.action_space.n,self.dic_phase_expansion)
                feature = np.concatenate([ob, feature_p], axis=1)
            else:
                feature = np.concatenate([ob, phase], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions = self.model(observation, train=True)
        actions = actions.clone().detach().numpy()
        return np.argmax(actions, axis=1)

    def sample(self):
        return np.random.randint(0, self.num_phases, self.sub_agents)

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def remember(self, last_ob, last_phase, action, reward, next_ob, cur_phase, key):
        self.replay_buffer.append((key, (last_ob, last_phase, action, reward, next_ob, cur_phase)))

    def _batchwise(self, samples):
        # (batch_size,12)
        obs_t_all=[item[1][0] for item in samples]
        obs_tp_all=[item[1][4] for item in samples]
        obs_t = [utils.remove_right_lane(x) for x in obs_t_all]
        obs_tp = [utils.remove_right_lane(x) for x in obs_tp_all]
        obs_t = np.concatenate(obs_t) # (batch,8)
        obs_tp = np.concatenate(obs_tp) # (batch,8)
        if self.phase:
            if self.one_hot:
                phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n, self.dic_phase_expansion) for item in samples])
                phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n, self.dic_phase_expansion) for item in samples])
            else:
                phase_t = np.concatenate([item[1][1] for item in samples])
                phase_tp = np.concatenate([item[1][5] for item in samples])
            feature_t = np.concatenate([obs_t, phase_t], axis=1) # (batch,16)
            feature_tp = np.concatenate([obs_tp, phase_tp], axis=1) # (batch,16)
        else:
            feature_t = obs_t
            feature_tp = obs_tp
        # (batch_size,16)
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
        out = self.target_model(b_tp, train=False) # (batch_size,8)
        target = rewards + self.gamma * torch.max(out, dim=1)[0] # (batch_size)
        target_f = self.model(b_t, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model(b_t, train=True), target_f)
        # for i in range(self.batch_size):
        #     out = self.target_model(next_input, train=False)
        #     target = reward + self.gamma * torch.max(out, dim=1)[0]
        #     target_f = self.model(input_list, train=False)
        #     target_f[0][action] = target[0]
        #     loss = self.criterion(self.model(input_list, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().numpy()

    def load_model(self, e):
        model_name = os.path.join(
            Registry.mapping['logger_mapping']['output_path'].path, 'model', f'{e}_{self.rank}.pt')
        self.model = FRAP(
            self.dic_agent_conf, self.dic_traffic_env_conf, self.num_actions, self.ob_length, self.num_phases)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = FRAP(
            self.dic_agent_conf, self.dic_traffic_env_conf, self.num_actions, self.ob_length, self.num_phases)
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(
            Registry.mapping['logger_mapping']['output_path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.model.state_dict(), model_name)


class FRAP(nn.Module):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, num_actions, ob_length, num_phases):
        super(FRAP, self).__init__()
        self.num_actions = num_actions
        self.ob_length = ob_length
        self.num_phases = num_phases
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf

        self.embedding_1 = nn.Embedding(2, 4)
        self.dense = nn.Linear(1, 4)
        self.lane_embedding = nn.Linear(8, 16)
        self.relation_embedd = nn.Embedding(2, 4)
        self.conv_feature = nn.Conv2d(
            in_channels=32, out_channels=self.dic_agent_conf.param["d_dense"], kernel_size=1)
        self.conv_relation = nn.Conv2d(
            in_channels=4, out_channels=self.dic_agent_conf.param["d_dense"], kernel_size=1)
        self.hidden_layer1 = nn.Conv2d(
            in_channels=20, out_channels=self.dic_agent_conf.param["d_dense"], kernel_size=1)
        self.hidden_layer2 = nn.Conv2d(
            in_channels=20, out_channels=1, kernel_size=1)

    def _forward(self, feature_list):
        '''
        feature_list:(batch_size,16)
            [:8]: lane_num_vehicle
            [8:]: cur_phase
        '''
        batch_size, _ = feature_list.shape
        p = F.sigmoid(self.embedding_1(feature_list[::,8:].long())) # (b,8,4)
        dic_lane = {}
        for i, m in enumerate(self.dic_traffic_env_conf.param["list_lane_order"]):
            tmp_vec = F.sigmoid(self.dense(feature_list[::,i:i+1])) # (b,4)
            dic_lane[m] = torch.cat([tmp_vec, p[:,i]],dim=1) # (b,8)
        if self.num_actions == 8:
            list_phase_pressure = []
            for phase in self.dic_traffic_env_conf.param["phases"]:
                m1, m2 = phase.split("_")
                tmp1 = F.relu(self.lane_embedding(dic_lane[m1])) #(b,16)
                tmp2 = F.relu(self.lane_embedding(dic_lane[m2])) #(b,16)
                list_phase_pressure.append(tmp1.add(tmp2))

        elif self.num_actions == 4:
            list_phase_pressure = []
            for phase in self.dic_traffic_env_conf.param["phases"]:
                m1, m2 = phase.split("_")
                list_phase_pressure.append(torch.cat([dic_lane[m1], dic_lane[m2]],dim=1))

        constant = self.relation(batch_size) # (b,8,7)
        relation_embedding = self.relation_embedd(constant.long()) # (b,8,7,4)

        # rotate the phase pressure
        if self.dic_agent_conf.param["rotation"]:
            list_phase_pressure_recomb = []
            num_phase = self.num_phases
            for i in range(num_phase):
                for j in range(num_phase):
                    if i != j:
                        list_phase_pressure_recomb.append(
                            torch.cat([list_phase_pressure[i], list_phase_pressure[j]],dim=1)) # (b,32)
            list_phase_pressure_recomb = (torch.stack(list_phase_pressure_recomb)).permute(1,0,2) # (b,56,32)

            feature_map = torch.reshape(list_phase_pressure_recomb, shape=(-1, 8, 7, 32)) #(b,8,7,32)
            lane_conv = F.relu(self.conv_feature(feature_map.permute(0, 3, 1, 2))) #(b,8,7,32)->(b,32,8,7)->(b,20,8,7)
            relation_embedding = relation_embedding.permute(0, 3, 1, 2) # (b,8,7,4)->(b,4,8,7)
            if self.dic_agent_conf.param["merge"] == "multiply":
                relation_conv = self.conv_relation(relation_embedding) # (b,20,8,7)
                combine_feature = lane_conv*relation_conv # (b,20,8,7)
            # TODO check two methods
            elif self.dic_agent_conf.param["merge"] == "concat":
                relation_conv = self.conv_relation(relation_embedding)
                combine_feature = torch.cat(lane_conv, relation_conv)
            elif self.dic_agent_conf.param["merge"] == "weight":
                relation_conv = self.conv_relation(relation_embedding)
                tmp_wei = (lambda x: x.repeat(1, 1, 5))(relation_conv)
                combine_feature = lane_conv*tmp_wei
            hidden_layer = F.relu(self.hidden_layer1(combine_feature)) # (b,20,8,7)
            before_merge = self.hidden_layer2(hidden_layer) # (b,1,8,7)
            before_merge = torch.reshape(before_merge, shape=(-1, 8, 7)) # (b,8,7)
            q_values = (lambda x: torch.sum(x, dim=2))(before_merge) # (b,8)
        return q_values

    def forward(self, feature_list, train=True):
        if train:
            return self._forward(feature_list)
        else:
            with torch.no_grad():
                return self._forward(feature_list)

    def relation(self, batch_size):
        relations = []
        for p1 in self.dic_traffic_env_conf.param["phases"]:
            zeros = [0, 0, 0, 0, 0, 0, 0]
            count = 0
            for p2 in self.dic_traffic_env_conf.param["phases"]:
                if p1 == p2:
                    continue
                m1 = p1.split("_")
                m2 = p2.split("_")
                if len(list(set(m1 + m2))) == 3:
                    zeros[count] = 1
                count += 1
            relations.append(zeros)
        relations = np.array(relations).reshape(8, 7)
        constant = torch.tensor(relations, dtype=torch.float32)
        constant = constant.repeat(batch_size, 1, 1)
        return constant
