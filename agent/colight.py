from . import RLAgent
from common.registry import Registry
import numpy as np
import os
import random
from collections import OrderedDict, deque
import gym

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator

import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

import pickle as pkl

@Registry.register_model('model')
class CoLightAgent(RLAgent):
    #  TODO: test multiprocessing effect on agent or need deep copy here
    def __init__(self, world, prefix):
        super().__init__(world)
        """
        multi-agent in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        self.buffer_size = Registry.mapping['task_mapping']['task_setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph
        self.world = world
        self.sub_agents = len(self.world.intersections)
        # TODO: support dynamic graph later
        self.edge_idx = torch.tensor(self.graph['sparse_adj'].T, dtype=torch.long)  # source -> target

        #  model parameters
        self.phase = Registry.mapping['world_mapping']['traffic_setting'].param['phase']
        self.one_hot = Registry.mapping['world_mapping']['traffic_setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['model_setting'].param

        #  get generator for CoLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # TODO: add irregular control of signals in the future
        self.action_space = gym.spaces.Discrete(len(self.world.intersections[0].phases))

        if self.phase:
            # TODO: irregular ob and phase in the future
            if self.one_hot:
                self.ob_length = self.ob_generator[0][1].ob_length + len(self.world.intersections[0].phases)
            else:
                self.ob_length = self.ob_generator[0][1].ob_length + 1
        else:
            self.ob_length = self.ob_generator[0][1].ob_length

        self.get_attention = Registry.mapping['logger_mapping']['logger_setting'].param['get_attention']
        # train parameters
        self.prefix = prefix
        self.gamma = Registry.mapping['model_mapping']['model_setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['model_setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['model_setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['model_setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['model_setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['model_setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['model_setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['model_setting'].param['batch_size']

        self.model = self._build_model()
        print(self.model)
        self.target_model = self._build_model()
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)

    def get_ob(self):
        x_obs = []  # sub_agents * lane_nums,
        for i in range(len(self.ob_generator)):
            x_obs.append((self.ob_generator[i][1].generate()) / self.vehicle_max)
        # construct edge information
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        # TODO: test output
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards

    def get_phase(self):
        # TODO: test phase output onehot/int
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = np.array(phase)
        return phase

    def get_reward_test(self):
        vehicle_reward = []
        vehicle_nums = self.world.get_info("lane_waiting_count")
        for i in range(self.sub_agents):
            node_id = self.graph["node_idx2id"][i]
            node_dict = self.world.id2intersection[node_id]
            nvehicles = 0
            tmp_vehicle = []
            in_lanes = []
            for road in node_dict.in_roads:
                from_zero = (road["startIntersection"] == node_dict.id) if self.world.RIGHT else (
                        road["endIntersection"] == node_dict.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))
            for lane in vehicle_nums.keys():
                if lane in in_lanes:
                    nvehicles += vehicle_nums[lane]
                    tmp_vehicle.append(vehicle_nums[lane])
            # vehicle_reward.append(-nvehicles)
            tmp_vehicle = np.array(tmp_vehicle)
            vehicle_reward.append(-tmp_vehicle.sum())  # return the average length of a intersection
        vehicle_reward = np.array(vehicle_reward)
        return vehicle_reward

    def get_action(self, ob, phase, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        observation = torch.tensor(ob, dtype=torch.float32)
        edge = self.edge_idx
        dp = Data(x=observation, edge_index=edge)
        # TODO: not phase not used

        if self.get_attention:
            # TODO: collect attention matrix later
            actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False)
            att = None
            actions = actions.clone().detach().numpy()
            return np.argmax(actions, axis=1), att  # [batch, agents], [batch, agents, nv, neighbor]
        else:
            actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False)
            actions = actions.clone().detach().numpy()
            return np.argmax(actions, axis=1)  # [batch, agents] TODO: check here

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def _build_model(self):
        model = ColightNet(self.ob_length, self.action_space.n, **self.model_dict)
        return model

    def remember(self, last_obs, last_phase, actions, rewards, obs, cur_phase, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        rewards = []
        for item in samples:
            dp = item[1]
            state = torch.tensor(dp[0], dtype=torch.float32)
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(dp[4], dtype=torch.float32)
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp[3])
            actions.append(dp[2])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here

        return batch_t, batch_tp, rewards, actions

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)

        out = self.target_model(x=b_tp.x, edge_index=b_tp.edge_index, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(x=b_t.x, edge_index=b_t.edge_index, train=False)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model(x=b_t.x, edge_index=b_t.edge_index, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model', f'{e}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class ColightNet(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(ColightNet, self).__init__()
        self.model_dict = kwargs
        self.action_space = gym.spaces.Discrete(output_dim)
        self.features = input_dim
        self.module_list = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(self.features, layers=self.model_dict.get('NODE_EMB_DIM'))
        for i in range(self.model_dict.get('N_LAYERS')):
            block = MultiHeadAttModel(d=self.model_dict.get('INPUT_DIM')[i],
                                      dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                      d_out=self.model_dict.get('OUTPUT_DIM')[i],
                                      nv=self.model_dict.get('NUM_HEADS')[i],
                                      suffix=i)
            self.module_list.append(block)
        output_dict = OrderedDict()

        if len(self.model_dict['OUTPUT_LAYERS']) != 0:
            # TODO: dubug this branch
            for l_idx, l_size in enumerate(self.model_dict['OUTPUT_LAYERS']):
                name = f'output_{l_idx}'
                if l_idx == 0:
                    h = nn.Linear(block.d_out, l_size)
                else:
                    h = nn.Linear(self.model_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
                output_dict.update({name: h})
                name = f'relu_{l_idx}'
                output_dict.update({name: nn.ReLU})
            out = nn.Linear(self.model_dict['OUTPUT_LAYERS'][-1], self.action_space.n)
        else:
            out = nn.Linear(block.d_out, self.action_space.n)
        name = f'output'
        output_dict.update({name: out})
        self.output_layer = nn.Sequential(output_dict)

    def forward(self, x, edge_index, train=True):
        h = self.embedding_MLP.forward(x, train)
        #TODO: implement att
        for mdl in self.module_list:
            h = mdl.forward(h, edge_index, train)
        if train:
            h = self.output_layer(h)
        else:
            with torch.no_grad():
                h = self.output_layer(h)
        return h


class Embedding_MLP(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP, self).__init__()
        constructor_dict = OrderedDict()
        for l_idx, l_size in enumerate(layers):
            name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                constructor_dict.update({name: h})
            else:
                h = nn.Linear(layers[l_idx - 1], l_size)
                constructor_dict.update({name: h})
            name = f"n_relu_{l_idx}"
            constructor_dict.update({name: nn.ReLU()})
        self.embedding_node = nn.Sequential(constructor_dict)

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class MultiHeadAttModel(MessagePassing):
    """
    inputs:
        In_agent [bacth,agent,128]
        In_neighbor [agent, neighbor_num]
        l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
        d: dimension of agent's embedding
        dv: dimension of each head
        dout: dimension of output
        nv: number of head (multi-head attention)
    output:
        -hidden state: [batch,agent,32]
        -attention: [batch,agent,neighbor]
    """
    def __init__(self, d, dv, d_out, nv, suffix):
        super(MultiHeadAttModel, self).__init__(aggr='add')
        self.d = d
        self.dv = dv
        self.d_out = d_out
        self.nv = nv
        self.suffix = suffix
        # target is center
        self.W_target = nn.Linear(d, dv * nv)
        self.W_source = nn.Linear(d, dv * nv)
        self.hidden_embedding = nn.Linear(d, dv * nv)
        self.out = nn.Linear(dv, d_out)
        self.att_list = []
        self.att = None

    def _forward(self, x, edge_index):
        # TODO: test batch is shared or not

        # x has shape [N, d], edge_index has shape [E, 2]
        edge_index, _ = add_self_loops(edge_index=edge_index)
        aggregated = self.propagate(x=x, edge_index=edge_index)  # [16, 16]
        out = self.out(aggregated)
        out = F.relu(out)  # [ 16, 128]
        #self.att = torch.tensor(self.att_list)
        return out

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)

    def message(self, x_i, x_j, edge_index):
        h_target = F.relu(self.W_target(x_i))
        h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)
        agent_repr = h_target.permute(1, 0, 2)

        h_source = F.relu(self.W_source(x_j))
        h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv)
        neighbor_repr = h_source.permute(1, 0, 2)   #[nv, E, dv]
        index = edge_index[1]  # which is target

        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [5, 64]
        max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [5, 16]
        max_i = max_node.index_select(1, index=index)  # [5, 64]
        ec_i = torch.add(e_i, -max_i)
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [5, 16]
        normst_node = torch.add(norm_node, 1e-12)  # [5, 16]
        normst_i = normst_node.index_select(1, index)  # [5, 64]

        alpha_i = ecexp_i / normst_i  # [5, 64]
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
        # alpha_i_expand = torch.permute(alpha_i_expand, (1, 2, 0))  # [5, 64, 16]
        alpha_i_expand = alpha_i_expand.permute(((1, 2, 0)))
        hidden_neighbor = F.relu(self.hidden_embedding(x_j))
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)

        # TODO: attention ouput in the future
        self.att_list.append(alpha_i)  # [64, 16]
        return out

    def get_att(self):
        if self.att is None:
            print('invalid att')
        return self.att
