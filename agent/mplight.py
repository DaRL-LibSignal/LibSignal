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

@Registry.register_model('mplight')
class MPLightAgent(RLAgent):
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

    def _build_model(self):
        model = FRAP(
            self.dic_agent_conf, self.dic_traffic_env_conf, self.num_actions, self.ob_length, self.num_phases)
        return model


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