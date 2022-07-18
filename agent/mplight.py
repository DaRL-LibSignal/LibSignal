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
from pfrl.q_functions import DiscreteActionValueHead
from agent.utils import SharedEpsGreedy
from pfrl import replay_buffers
from typing import Any, Sequence
from pfrl.agents import DQN
from pfrl.utils.contexts import evaluating
from agent import utils

'''MPLight is default set Shared Agent'''

@Registry.register_model('mplight')
class MPLightAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world,world.intersection_ids[rank])
        self.dic_agent_conf = Registry.mapping['model_mapping']['setting']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['setting']
        self.dic_trainer_conf = Registry.mapping['trainer_mapping']['setting']
        
        self.gamma = self.dic_agent_conf.param["gamma"]
        self.grad_clip = self.dic_agent_conf.param["grad_clip"]
        # self.epsilon = self.dic_agent_conf.param["epsilon"]
        # self.epsilon_min = self.dic_agent_conf.param["epsilon_min"]
        # self.epsilon_decay = self.dic_agent_conf.param["epsilon_decay"]
        self.learning_rate = self.dic_agent_conf.param["learning_rate"]
        self.batch_size = self.dic_agent_conf.param["batch_size"]
        self.num_phases = len(self.dic_traffic_env_conf.param["phases"])
        self.num_actions = len(self.dic_traffic_env_conf.param["phases"])
        self.buffer_size = self.dic_trainer_conf.param['buffer_size']
        # self.replay_buffer = deque(maxlen=self.buffer_size)
        self.replay_buffer = replay_buffers.ReplayBuffer(self.buffer_size)
        

        self.world = world
        self.rank = rank
        self.sub_agents = len(self.world.intersections)
        self.inter_id = self.world.intersection_ids[self.rank]
        self.phase = self.dic_traffic_env_conf.param['phase']
        self.one_hot = self.dic_traffic_env_conf.param['one_hot']
        self.action_space_list = [gym.spaces.Discrete(len(x.phases)) for x in self.world.intersections]
        # create competition matrix
        map_name = self.dic_traffic_env_conf.param['dataset_name']
        self.phase_pairs = self.dic_traffic_env_conf.param['signal_config'][map_name]['phase_pairs']
        self.comp_mask = self.relation()
        self.valid_acts = self.dic_traffic_env_conf.param['signal_config'][map_name]['valid_acts']
        self.reverse_valid = None
        self.model = None
        self.optimizer = None
        episodes = self.dic_trainer_conf.param['episodes'] * 0.8
        steps = self.dic_trainer_conf.param['steps']
        action_interval = self.dic_trainer_conf.param['action_interval']
        total_steps = episodes * steps / action_interval
        self.explorer = SharedEpsGreedy(
                self.dic_agent_conf.param["eps_start"],
                self.dic_agent_conf.param["eps_end"],
                self.sub_agents*total_steps,
                lambda: np.random.randint(len(self.phase_pairs)),
            )
        self.agents_iner = self._build_model()

        # get generators for MPLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  get phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, node_obj, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        #  get queue generator
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_waiting_count"], 
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        #  get delay generator
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_delay"], 
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        # TODO check ob_length, compared with presslight and original mplight and RESCO-mplight
        # this is extracted from presslight so far
        # if self.phase:
        #     if self.one_hot:
        #         self.ob_length = self.ob_generator.ob_length + len(self.inter_obj.phases) # 32
        #     else:
        #         self.ob_length = self.ob_generator.ob_length + 1 # 25
        # else:
        #     self.ob_length = self.ob_generator.ob_length # 24

    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, node_obj, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        queues = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_waiting_count"], 
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        delays = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_delay"], 
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        sorted(delays, key=lambda x: x[0])
        self.delay = delays
    
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

    def get_ob(self):
        """
        output: [sub_agents,lane_nums]
        """
        x_obs = []  # sub_agents * lane_nums,
        for i in range(len(self.ob_generator)):
            x_obs.append((self.ob_generator[i][1].generate()))
        # x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        # TODO: test output
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards))
        return rewards

    def get_phase(self):
        """
        output: [sub_agents,]
        """
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_queue(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        queue = []
        for i in range(len(self.queue)):
            queue.append((self.queue[i][1].generate()))
        tmp_queue = np.squeeze(np.array(queue))
        if self.sub_agents == 1:
            queue = np.sum(tmp_queue)
        else:
            queue = [np.sum(x) for x in tmp_queue]
        # queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape)==1 else 0)
        return queue # [intersections,]

    def get_delay(self):
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay))
        return delay # [intersections,]

    def get_action(self, ob, phase, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length]
        :param phase: [agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [agents] -> action taken by environment
        """
        # concate ob and phase
        if self.phase:
            if self.one_hot:
                obs = np.concatenate([utils.idx2onehot(phase, self.action_space.n), ob], axis=1)
            else:
                obs = [np.concatenate([np.array([phase[i]]), ob[i]]) for i in range(self.sub_agents)]
        else:
            obs = ob

        if self.reverse_valid is None and self.valid_acts is not None:
            self.reverse_valid = dict()
            for signal_id in self.valid_acts:
                self.reverse_valid[signal_id] = {v: k for k, v in self.valid_acts[signal_id].items()}

        batch_obs = obs
        if self.valid_acts is None:
            batch_valid = None
            batch_reverse = None
        else:
            # TODO debug
            batch_valid = [self.valid_acts.get(agent_id) for agent_id in
                           ob.keys()]
            batch_reverse = [self.reverse_valid.get(agent_id) for agent_id in
                          ob.keys()]
        batch_acts = self.agents_iner.act(batch_obs,
                                valid_acts=batch_valid,
                                reverse_valid=batch_reverse, test=test)
        acts = np.array(batch_acts)
        return acts
        

    def sample(self):
        pass
        # """Applicable to various traffic light plans at each intersection."""
        # ran_phase = []
        # for x in self.action_space_list:
        #     ran_phase.append(np.random.randint(0, x.n))
        # return ran_phase

    def remember(self, last_obs, last_phase, actions, rewards, obs, cur_phase, key):
        """pfrl.dqn have automatically implemented this part, so there is no need to implement it."""
        pass
    
    def do_observe(self, ob, phase, reward, done):
        # TODO how to handle inregular lane length?
        # # padding ob length
        # max_len = max([len(x) for x in ob])
        # ob_uni = list(map(lambda l:list(l) + [0.]*(max_len - len(l)), ob))
        # last_ob_uni = list(map(lambda l:list(l) + [0.]*(max_len - len(l)), last_ob))

        # concate ob and phase
        if self.phase:
            if self.one_hot:
                obs = np.concatenate([utils.idx2onehot(phase, self.action_space.n), ob], axis=1)
            else:
                obs = [np.concatenate([np.array([phase[i]]), ob[i]]) for i in range(self.sub_agents)]
        else:
            obs = ob
        reset = [False] * self.sub_agents
        dones = [done] * self.sub_agents
        rewards = [reward] if isinstance(reward, float) else reward
        self.agents_iner.observe(obs, rewards, dones, reset)
                
    def _build_model(self):
        self.model = FRAP(self.dic_agent_conf, self.dic_traffic_env_conf, self.num_actions, self.phase_pairs, self.comp_mask)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        agents = MPLight_InerAgent(self.model, self.optimizer, self.replay_buffer, self.gamma, self.explorer,
                        minibatch_size=self.batch_size, replay_start_size=self.batch_size, 
                        phi=lambda x: np.asarray(x, dtype=np.float32),
                        # TODO check what is TARGET_UPDATE, target_update_interval, update_interval
                        target_update_interval=self.dic_agent_conf.param["target_update"]*self.sub_agents, update_interval=self.sub_agents
                        )
        return agents

    def update_target_network(self):
        pass

    def train(self):
        result = self.agents_iner.get_statistics()
        return result[1][1]


    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        
        self.agents_iner = self._build_model()
        # self.agents_iner.load_state_dict(torch.load(model_name))
        tmp_dict = {}
        tmp_dict.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        # torch.save(self.target_model.state_dict(), model_name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_name)

class MPLight_InerAgent(DQN):
    def __init__(self, q_function, optimizer,replay_buffer, 
                 gamma, explorer, minibatch_size, replay_start_size, phi, 
                 target_update_interval, update_interval):
        super().__init__(q_function, optimizer, replay_buffer, gamma, explorer,
                         minibatch_size=minibatch_size, replay_start_size=replay_start_size, phi=phi,
                         target_update_interval=target_update_interval, update_interval=update_interval)
        # self.batch_last_state = None
        # self.batch_last_action = None

    def act(self, obs, valid_acts=None, reverse_valid=None, test=False):
        return self.batch_act(obs, valid_acts=valid_acts, reverse_valid=reverse_valid, test=test)

    def observe(self, obs, reward, done, reset):
        # self.training = True
        self.batch_observe(obs, reward, done, reset)

    def batch_act(self, batch_obs, valid_acts=None, reverse_valid=None, test=False):
        # self.training = not test
        if valid_acts is None: 
            return super(MPLight_InerAgent, self).batch_act(batch_obs)
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            # TODO do not use gpu
            batch_qvals = batch_av.params[0].detach().cpu().numpy() # shape: [sub_agents, 1+ob_length(remove right_lane?)]
            batch_argmax = []
            for i in range(len(batch_obs)):
                batch_item = batch_qvals[i]
                max_val, max_idx = None, None
                for idx in valid_acts[i]:
                    batch_item_qval = batch_item[idx]
                    if max_val is None:
                        max_val = batch_item_qval
                        max_idx = idx
                    elif batch_item_qval > max_val:
                        max_val = batch_item_qval
                        max_idx = idx
                batch_argmax.append(max_idx)
            batch_argmax = np.asarray(batch_argmax)

        if self.training:
            batch_action = []
            for i in range(len(batch_obs)):
                av = batch_av[i : i + 1]
                greed = batch_argmax[i]
                act, greedy = self.explorer.select_action(self.t, lambda: greed, action_value=av, num_acts=len(valid_acts[i]))
                if not greedy:
                    act = reverse_valid[i][act]
                batch_action.append(act)

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax

        valid_batch_action = []
        for i in range(len(batch_action)):
            valid_batch_action.append(valid_acts[i][batch_action[i]])
        return valid_batch_action        

        
class FRAP(nn.Module):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, output_shape, phase_pairs, competition_mask):
        super(FRAP, self).__init__()
        self.oshape = output_shape
        self.phase_pairs = phase_pairs
        self.comp_mask = competition_mask
        self.demand_shape = dic_agent_conf.param['demand_shape']      # Allows more than just queue to be used
        self.one_hot = dic_traffic_env_conf.param['one_hot']
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

        self.head = DiscreteActionValueHead()

    def forward(self, states):
        '''
        :params states: [agents, ob_length]
        ob_length:concat[len(one_phase),len(intersection_lane)]
        '''
        num_movements = num_movements = int((states.size()[1]-1)/self.demand_shape) if not self.one_hot else int((states.size()[1]-len(self.phase_pairs))/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, 0].to(torch.int64) if not self.one_hot else states[:, :len(self.phase_pairs)].to(torch.int64)
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
        for i in range(num_movements):
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
        q_values = torch.sum(combine_features, dim=-1)
        return self.head(q_values)
