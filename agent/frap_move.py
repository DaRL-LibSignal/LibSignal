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
from copy import deepcopy


@Registry.register_model('frap_move')
class FRAP_MOVE_DQNAgent(RLAgent):
    '''
    FRAP_MOVE_DQNAgent consists of FRAP and methods for training agents, communicating with environment, etc.
    '''
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
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1
        self.rank = rank

        self.phase = self.dic_agent_conf.param['phase']
        assert self.phase is True
        self.one_hot = self.dic_agent_conf.param['one_hot']
        assert self.one_hot is False

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

        map_name = self.dic_traffic_env_conf.param['network']

        # set valid action
        # all_valid_acts = self.dic_traffic_env_conf.param['signal_config'][map_name]['valid_acts']
        # if all_valid_acts is None:
        #     self.valid_acts = None
        # else:
        #     if self.inter_id in all_valid_acts.keys():
        #         self.inter_name = self.inter_id
        #     else:
        #         if 'GS_' in self.inter_id:
        #             self.inter_name = self.inter_id[3:]
        #         else:
        #             self.inter_name = 'GS_' + self.inter_id
        #     self.valid_acts = all_valid_acts[self.inter_name]
        self.inter_name = self.inter_id
        if 'GS_' in self.inter_id:
            self.inter_name = self.inter_id[3:]
        
        # self.ob_order = None
        # if 'lane_order' in self.dic_traffic_env_conf.param['signal_config'][map_name].keys():
        #     self.ob_order = self.dic_traffic_env_conf.param['signal_config'][map_name]['lane_order'][self.inter_name]

        # set phase_pairs
        # self.phase_pairs = []
        # all_phase_pairs = self.dic_traffic_env_conf.param['signal_config'][map_name]['phase_pairs']
        # for idx in self.valid_acts:
        #     self.phase_pairs.append([self.ob_order[x] for x in all_phase_pairs[idx]])
        

        # self.comp_mask = self.relation()
        # self.num_phases = len(self.phase_pairs)
        # self.num_actions = len(self.phase_pairs)
        
        self.lane_names = []
        [self.lane_names.extend(l) for l in self.ob_generator.lanes]

        self.directions = self.ob_generator.directions
        self.road_names = self.ob_generator.roads
        # we take mapping strategy that assigning roads direction to the most closed ESWN direction
        self.movements = [self._orthogonal_mapping(rad) for rad in self.directions]
        self.twelve_movements = ['N_L', 'N_T', 'N_R','E_L', 'E_T', 'E_R', 'S_L', 'S_T', 'S_R', 'W_L', 'W_T', 'W_R']
        self.inter_info = [self.world.roadnet['intersections'][idx] for idx, i in enumerate(self.world.roadnet['intersections']) if i['id'] == self.inter_id][0]
        self.linkage_movement = {(i['startRoad'], i['endRoad']): i['type'] for i in self.inter_info['roadLinks']}
        self.lane2movements = self._construct_lane2movement_mapping()
        self.phase2movements = self._phase_avail_movements()
        self.comp_mask = self.relation()
        self.phase2movements = torch.tensor(self.phase2movements)
        self.num_phases = self.phase2movements.shape[0]
        self.num_actions = self.phase2movements.shape[0]

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.criterion = nn.MSELoss(reduction='mean')

    def __repr__(self):
        return self.model.__repr__()

    def _construct_lane2movement_mapping(self):
        result = np.zeros([len(self.lane_names), len(self.twelve_movements)])
        mapping = self.inter_info['roadLinks']
        for r_link in mapping:
            tp = r_link['type']
            if tp == 'turn_left':
                #  only work for atlanta now, remove U-turn
                start_r = r_link['startRoad'].split('#')[0].replace('-', '')
                end_r = r_link['endRoad'].split('#')[0].replace('-', '')
                if start_r == end_r:
                    continue
                turn = 'L'
            elif tp == 'go_straight':
                turn = 'T'
            elif tp == 'turn_right':
                turn = 'R'
            else:
                raise ValueError
            for l_link in r_link['laneLinks']:
                idx = l_link['startLaneIndex']
                r_idx = self.lane_names.index(r_link['startRoad']+'_'+str(idx))
                c_idx = self.twelve_movements.index(self.movements[r_idx] + '_'+turn)
                result[r_idx, c_idx] = 1
        return result

    def _orthogonal_mapping(self, rad):
        if  rad > 5.49779 or rad < 0.785398:
            return 'N'
        elif rad >=0.785398 and rad < 2.35619:
            return 'E'
        elif rad >= 2.35619 and rad < 3.92699:
            return 'S'
        elif rad >= 3.92699 and rad < 5.49779:
            return 'W'
        else:
            raise ValueError

    def _phase_avail_movements(self):
        # no yellow phase
        result = np.zeros([self.action_space.n, len(self.twelve_movements)])
        for p in range(self.action_space.n):
            avail_road_links_id = self.ob_generator.I.phase_available_roadlinks[p]
            for l in avail_road_links_id:
                linkage = self.ob_generator.I.roadlinks[l]
                start = linkage[0]
                end = linkage[1]
                tp = self.linkage_movement[(start, end)]
                if tp == 'turn_left':
                    #  only work for atlanta now, remove U-turn
                    start_r = start.split('#')[0].replace('-', '')
                    end_r = end.split('#')[0].replace('-', '')
                    if start_r == end_r:
                        continue
                    turn = 'L'
                elif tp == 'go_straight':
                    turn = 'T'
                elif tp == 'turn_right':
                    turn = 'R'
                d = self.movements[self.road_names.index(start)]
                direction = self.twelve_movements.index(d + "_" + turn)
                result[p, direction] = 1
        return result

    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
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
        '''
        relation
        Get the phase competition relation between traffic movements.

        :param: None
        :return comp_mask: matrix of phase competition relation
        '''
        comp_mask = []
        # remove connection at all phase, then compute if there is a same connection here
        removed_phase2movements = deepcopy(self.phase2movements)
        removed_phase2movements[:, np.sum(self.phase2movements, axis=0) == self.phase2movements.shape[0]] = 0
        for i in range(self.phase2movements.shape[0]):
            zeros = np.zeros(self.phase2movements.shape[0] - 1, dtype=np.int)
            cnt = 0
            for j in range(self.phase2movements.shape[0]):
                if i == j: continue

                pair_a = removed_phase2movements[i]
                pair_b = removed_phase2movements[j]
                if np.dot(pair_a, pair_b) >= 1: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = torch.from_numpy(np.asarray(comp_mask))
        return comp_mask 

    def _build_model(self):
        '''
        _build_model
        Build a FRAP agent.

        :param: None
        :return model: FRAP model
        '''
        model = FRAP_move(self.dic_agent_conf, self.num_actions, self.phase2movements, self.comp_mask)
        return model

    def get_ob(self):
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        tmp = self.ob_generator.generate()
        return np.dot(tmp, self.lane2movements)

    def get_reward(self):
        '''
        get_reward
        Get reward from environment.

        :param: None
        :return rewards: rewards generated by reward_generator
        '''
        rewards = []
        rewards.append(self.reward_generator.generate())
        # TODO check whether to multiply 12
        rewards = np.squeeze(np.array(rewards))# * self.num_phases
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
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        '''
        get_action
        Generate action.

        :param ob: observation, the shape is (1,12)
        :param phase: current phase, the shape is (1,)
        :param test: boolean, decide whether is test process
        :return: action that has the highest score
        '''
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        if self.phase2movements.shape[0] == 1:
            return np.array([0])

        feature = np.concatenate([phase.reshape(1,-1), ob.reshape(1,-1)], axis=1)
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
        # return np.random.randint(0, self.num_phases, self.sub_agents)
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def update_target_network(self):
        '''
        update_target_network
        Update params of target network.

        :param: None
        :return: None
        '''
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

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
        self.replay_buffer.append((key, (last_obs.reshape(1,-1), last_phase, actions, rewards, obs.reshape(1,-1), cur_phase.reshape(1,-1))))

    def _batchwise(self, samples):
        '''
        _batchwise
        Reconstruct the samples into batch form(last state, current state, reward, action).

        :param samples: original samples record in replay buffer
        :return state_t, state_tp, rewards, actions: information with batch form
        '''
        # (batch_size,12)
        obs_t_all=[item[1][0] for item in samples] # last_obs(batch, 1, lane_num)
        obs_tp_all=[item[1][4] for item in samples] # cur_obs
        # obs_t = [utils.remove_right_lane(x) for x in obs_t_all]
        # obs_tp = [utils.remove_right_lane(x) for x in obs_tp_all]
        obs_t = obs_t_all
        obs_tp = obs_tp_all
        obs_t = np.concatenate(obs_t) # (batch,lane_num)
        obs_tp = np.concatenate(obs_tp) # (batch,lane_num)

        phase_t = np.concatenate([item[1][1].reshape(1,-1) for item in samples]) # (batch, 1)
        phase_tp = np.concatenate([item[1][5].reshape(1,-1) for item in samples])
        feature_t = np.concatenate([phase_t, obs_t], axis=1) # (batch,ob_length)
        feature_tp = np.concatenate([phase_tp, obs_tp], axis=1)
        # (batch_size, ob_length)

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
        if len(self.replay_buffer) < self.batch_size:
            return
        if self.action_space.n == 1:
            return np.array(0)
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
        '''
        load_model
        Load model params of an episode.

        :param e: specified episode
        :return: None
        '''
        model_name = os.path.join(
            Registry.mapping['logger_mapping']['output_path'].path, 'model', f'{e}_{self.rank}.pt')
        self.model = FRAP_move(self.dic_agent_conf, self.num_actions, self.num_actions, self.phase2movements, self.comp_mask)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = FRAP_move(self.dic_agent_conf, self.num_actions, self.num_actions, self.phase2movements, self.comp_mask)
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


class FRAP_move(nn.Module):
    '''
    FRAP captures the phase competition relation between traffic movements through a modified network structure.
    '''
    def __init__(self, dic_agent_conf, output_shape, phase2movements, competition_mask):
        super(FRAP_move, self).__init__()
        self.oshape = output_shape
        self.phase2movements = phase2movements.float()
        self.comp_mask = competition_mask
        self.demand_shape = dic_agent_conf.param['demand_shape']      # Allows more than just queue to be used
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
        states: [agents, ob_length]
        ob_length:concat[len(one_phase),len(intersection_lane)]
        '''
        # if lane_num=12,then num_movements=12, but turning right do not be used
        num_movements = int((states.size()[1]-1)/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, :1].to(torch.int64)
        states = states[:, 1:].unsqueeze(1).repeat(1,self.phase2movements.shape[0], 1)
        states = states.float()
        
        # Expand action index to mark demand input indices
        extended_acts = []
        # for i in range(batch_size):
        # # index of phase
        #     act_idx = acts[i]
        #     connectivity = self.phase2movements[act_idx]
        #     extended_acts = torch.stack(connectivity)
        # phase_embeds = torch.sigmoid(self.p(extended_acts))

        connectivity = self.phase2movements[acts].to(torch.int64)
        phase_embeds = torch.sigmoid(self.p(connectivity)) # [B, 4, 3, 12]



        # if num_movements == 12:
        #     order_lane = [0,1,3,4,6,7,9,10] # remove turning_right phase
        # else:
        #     order_lane = [i for i in range(num_movements)]
        # for idx, i in enumerate(order_lane):

        # for i in range(num_movements):
        #     # phase = phase_embeds[:, idx]  # size 4
        #     phase = phase_embeds[:, i]  # size 4
        #     demand = states[:, i:i+self.demand_shape]
        #     demand = torch.sigmoid(self.d(demand))    # size 4
        #     phase_demand = torch.cat((phase, demand), -1)
        #     phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
        #     phase_demands.append(phase_demand_embed)
        # phase_demands = torch.stack(phase_demands, 1)

        all_phase_demand = states * self.phase2movements # [B, 3, 12] - checked
        all_phase_demand = torch.sigmoid(self.d(all_phase_demand.unsqueeze(-1))) # [B, 3, 12, 4] - checked
        phase_demand = torch.cat((all_phase_demand, phase_embeds.repeat(1,self.phase2movements.shape[0],1,1)), -1) # B, 3, 12, 8]
        phase_demand_embed = F.relu(self.lane_embedding(phase_demand) ) # [B, 3, 12, 16]
        phase_demand_agg = torch.sum(phase_demand_embed, dim=2) # [B, 3, 16]
        rotated_phases = []
        for i in range(phase_demand_agg.shape[-2]):
            for j in range(phase_demand_agg.shape[-2]):
                if i != j: rotated_phases.append(torch.cat((phase_demand_agg[:,i,:], phase_demand_agg[:,j,:]), -1))
        rotated_phases = torch.stack(rotated_phases, 1) # [B, 2*3, 32]
        rotated_phases = torch.reshape(rotated_phases,
                                       (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units)) # [B, 3, 2, 32]
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # [B, 32, 3, 2]
        rotated_phases = F.relu(self.lane_conv(rotated_phases)) # [B, 20, 3, 2]
        # pairs = []
        # for pair in self.phase_pairs:
        #     pairs.append(phase_demands[:, pair[0]] + phase_demands[:, pair[1]])
        


        # rotated_phases = []
        # for i in range(len(pairs)):
        #     for j in range(len(pairs)):
        #         if i != j: rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        # rotated_phases = torch.stack(rotated_phases, 1)
        # rotated_phases = torch.reshape(rotated_phases,
        #                                (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units)) # [B, 3, 2, 16]
        # rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        # rotated_phases = F.relu(self.lane_conv(rotated_phases))  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1)) # [B, 3, 2]
        relations = F.relu(self.relation_embedding(competition_mask)) # [B, 3, 2, 4] ? 
        relations = relations.permute(0, 3, 1, 2)  # [B, 4, 3, 2]
        relations = F.relu(self.relation_conv(relations))  # [B, 20, 3, 2]

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # # [B, 1, 3, 2]

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1)) # [B, 3, 2]
        q_values = (lambda x: torch.sum(x, dim=2))(combine_features) # (B, 3)
        return q_values
        

    def forward(self, states, train=True):
        if train:
            return self._forward(states)
        else:
            with torch.no_grad():
                return self._forward(states)
