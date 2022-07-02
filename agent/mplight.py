from . import RLAgent, SharedDQN
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

'''MPLight is default set Shared Agent'''

@Registry.register_model('mplight')
class MPLightAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world,world.intersection_ids[rank])
        self.dic_agent_conf = Registry.mapping['model_mapping']['model_setting']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['traffic_setting']
        
        self.gamma = self.dic_agent_conf.param["gamma"]
        self.grad_clip = self.dic_agent_conf.param["grad_clip"]
        # self.epsilon = self.dic_agent_conf.param["epsilon"]
        # self.epsilon_min = self.dic_agent_conf.param["epsilon_min"]
        # self.epsilon_decay = self.dic_agent_conf.param["epsilon_decay"]
        self.learning_rate = self.dic_agent_conf.param["learning_rate"]
        self.batch_size = self.dic_agent_conf.param["batch_size"]
        self.num_phases = len(self.dic_traffic_env_conf.param["phases"])
        self.num_actions = len(self.dic_traffic_env_conf.param["phases"])
        self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        # self.replay_buffer = deque(maxlen=self.buffer_size)
        self.replay_buffer = replay_buffers.ReplayBuffer(self.buffer_size)
        self.dic_trainer_conf = Registry.mapping['trainer_mapping']['trainer_setting']

        self.world = world
        self.sub_agents = len(self.world.intersections)
        # create competition matrix
        # map_name = self.world.intersections[0].map_name
        # TODO how to get map_name dynamically
        map_name = 'cologne3'
        self.phase_pairs = self.dic_traffic_env_conf.param['signal_config'][map_name]['phase_pairs']
        self.comp_mask = self.relation()
        self.valid_acts = self.dic_traffic_env_conf.param['signal_config'][map_name]['valid_acts']
        self.reverse_valid = None
        self.model = None
        self.optimizer = None
        episodes = Registry.mapping['trainer_mapping']['trainer_setting'].param['episodes']
        steps = Registry.mapping['trainer_mapping']['trainer_setting'].param['steps']
        action_interval = Registry.mapping['trainer_mapping']['trainer_setting'].param['action_interval']
        total_steps = episodes * steps / action_interval
        self.explorer = SharedEpsGreedy(
            # TODO check what does those params mean
                self.dic_agent_conf.param["eps_start"],
                self.dic_agent_conf.param["eps_end"],
                self.sub_agents*total_steps,
                lambda: np.random.randint(len(self.phase_pairs)),
            )
        self.agent = self._build_model()
    
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
        comp_mask = np.asarray(comp_mask)
        return comp_mask


    def reset(self):
        for ag in self.agents:
            ag.reset()

    def get_ob(self):
        """
        output: [sub_agents,lane_nums]
        """
        x_obs = []  # sub_agents * lane_nums,
        for ag in self.agents:
            x_obs.append(ag.ob_generator.generate())
        # x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        rewards = []  # sub_agents
        for ag in self.agents:
            rewards.append(ag.reward_generator.generate())
        # rewards = np.squeeze(np.array(rewards)) * 12
        return rewards

    def get_phase(self):
        """
        output: [sub_agents,]
        """
        phase = []  # sub_agents
        for ag in self.agents:
            phase.append(ag.phase_generator.generate())
        # phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_queue(self):
        queue = []
        for ag in self.agents:
            queue.append(ag.queue_generator.generate())
        # tmp_queue = np.squeeze(np.array(queue))
        # queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape)==2 else 0)
        return queue

    def get_delay(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        delay = []
        for ag in self.agents:
            delay.append(ag.delay_generator.generate())
        # delay = np.squeeze(np.array(delay))
        return delay # [intersections,]

    def get_action(self, ob, phase, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        # In RESCO, observation: 13=1(phase)+12(lanes)
        if self.reverse_valid is None and self.valid_acts is not None:
            self.reverse_valid = dict()
            for signal_id in self.valid_acts:
                self.reverse_valid[signal_id] = {v: k for k, v in self.valid_acts[signal_id].items()}

        batch_obs = [ob[agent_id] for agent_id in ob.keys()]
        if self.valid_acts is None:
            batch_valid = None
            batch_reverse = None
        else:
            batch_valid = [self.valid_acts.get(agent_id) for agent_id in
                           ob.keys()]
            batch_reverse = [self.reverse_valid.get(agent_id) for agent_id in
                          ob.keys()]
        batch_acts = self.agent.act(batch_obs,
                                valid_acts=batch_valid,
                                reverse_valid=batch_reverse)
        acts = dict()
        for i, agent_id in enumerate(ob.keys()):
            acts[agent_id] = batch_acts[i]
        return acts
        

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def _build_model(self):
        self.model = FRAP(self.dic_agent_conf, self.num_actions, self.phase_pairs, self.comp_mask)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.agent = SharedDQN(self.model, self.optimizer, self.replay_buffer, self.gamma, self.explorer,
                        minibatch_size=self.batch_size, replay_start_size=self.batch_size, 
                        phi=lambda x: np.asarray(x, dtype=np.float32),
                        # TODO check what is TARGET_UPDATE, target_update_interval, update_interval
                        target_update_interval=self.dic_agent_conf.param["target_update"]*self.sub_agents, update_interval=self.sub_agents
                        )

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        self.agent.observe()


    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)

class MPLight_SUBAgent(object):
    def __init__(self, world, inter_id):
        self.phase = Registry.mapping['world_mapping']['traffic_setting'].param['phase']
        self.one_hot = Registry.mapping['world_mapping']['traffic_setting'].param['one_hot']
        self.world = world
        # get generator for each Agent
        self.inter_id = inter_id
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj,
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)
        # TODO check ob_length, compared with presslight and original mplight and RESCO-mplight
        # this is extracted from presslight so far
        if self.phase:
            if self.one_hot:
                self.ob_length = self.ob_generator.ob_length + len(self.inter_obj.phases) # 32
            else:
                self.ob_length = self.ob_generator.ob_length + 1 # 25
        else:
            self.ob_length = self.ob_generator.ob_length # 24

    def reset(self):
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"], average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["pressure"], average="all", negative=True)
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)

        
class FRAP(nn.Module):
    def __init__(self, dic_agent_conf, output_shape, phase_pairs, competition_mask):
        super(FRAP, self).__init__()
        self.oshape = output_shape
        self.phase_pairs = phase_pairs
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

        self.head = DiscreteActionValueHead()

    def forward(self, states):
        '''
        :params states: [agents, ob_length]
        In RESCO, ob_length=13=1(phase)+12(vehicle_lane_level)
        '''
        num_movements = int((states.size()[1]-1)/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, 0].to(torch.int64)
        states = states[:, 1:]
        states = states.float()

        # Expand action index to mark demand input indices
        extended_acts = []
        for i in range(batch_size):
            act_idx = acts[i]
            pair = self.phase_pairs[act_idx]
            zeros = torch.zeros(num_movements, dtype=torch.int64)
            zeros[pair[0]] = 1
            zeros[pair[1]] = 1
            extended_acts.append(zeros)
        extended_acts = torch.stack(extended_acts)
        phase_embeds = torch.sigmoid(self.p(extended_acts))

        phase_demands = []
        for i in range(num_movements):
            phase = phase_embeds[:, i]  # size 4
            demand = states[:, i:i+self.demand_shape]
            demand = torch.sigmoid(self.d(demand))    # size 4
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands = torch.stack(phase_demands, 1)

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
        competition_mask = self.comp_mask.tile((batch_size, 1, 1))
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