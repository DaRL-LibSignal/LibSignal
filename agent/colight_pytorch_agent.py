from . import RLAgent
import random
import numpy as np
from collections import deque, OrderedDict
import os
import pickle
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch
import torch_scatter


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
    def __init__(self, d=128, dv=16, d_out=128, nv=8, suffix=-1):
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
        #TODO: confirm its a vector of size E
        # method 1: e_i = torch.einsum()
        # method 2: e_i = torch.bmm()
        # method 3: e_i = (a * b).sum(-1)
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
        alpha_i_expand = torch.permute(alpha_i_expand, (1, 2, 0))  # [5, 64, 16]
        # TODO: test x_j or x_i here -> should be x_j
        hidden_neighbor = F.relu(self.hidden_embedding(x_j))
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)

        # TODO: maybe here
        self.att_list.append(alpha_i)  # [64, 16]
        return out
    """
    def aggregate(self, inputs, edge_index):
        out = inputs
        index = edge_index[1]
    """




    def get_att(self):
        if self.att is None:
            print('invalid att')
        return self.att


class ColightNet(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(ColightNet, self).__init__()
        self.constructor_dict = kwargs
        self.action_space = self.constructor_dict.get('action_space') or 8
        self.modulelist = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(input_dim, layers=self.constructor_dict.get('NODE_EMB_DIM') or [128, 128])
        for i in range(self.constructor_dict.get('N_LAYERS')):
            module = MultiHeadAttModel(d=self.constructor_dict.get('INPUT_DIM')[i],
                                       dv=self.constructor_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                       d_out=self.constructor_dict.get('OUTPUT_DIM')[i],
                                       nv=self.constructor_dict.get('NUM_HEADS')[i],
                                       suffix=i)
            self.modulelist.append(module)
        output_dict = OrderedDict()

        """
        if self.constructor_dict.get('N_LAYERS') == 0:
            out = nn.Linear(128, self.action_space.n)
            name = f'output'
            output_dict.update({name: out})
            self.output_layer = nn.Sequential(output_dict)
        """
        output_dict = OrderedDict()
        if len(self.constructor_dict['OUTPUT_LAYERS']) != 0:
            # TODO: dubug this branch
            for l_idx, l_size in enumerate(self.constructor_dict['OUTPUT_LAYERS']):
                name = f'output_{l_idx}'
                if l_idx == 0:
                    h = nn.Linear(module.d_out, l_size)
                else:
                    h = nn.Linear(self.output_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
                output_dict.update({name: h})
                name = f'relu_{l_idx}'
                output_dict.update({name: nn.ReLU})
            out = nn.Linear(self.constructor_dict['OUTPUT_LAYERS'][-1], self.action_space.n)
        else:
            out = nn.Linear(module.d_out, self.action_space.n)
        name = f'output'
        output_dict.update({name: out})
        self.output_layer = nn.Sequential(output_dict)

    def forward(self, x, edge_index, train=True):
        h = self.embedding_MLP.forward(x, train)
        #TODO: implement att
        for mdl in self.modulelist:
            h = mdl.forward(h, edge_index, train)
        if train:
            h = self.output_layer(h)
        else:
            with torch.no_grad():
                h = self.output_layer(h)
        return h


class CoLightAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, world, traffic_env_conf, graph_setting, args):
        super().__init__(action_space, ob_generator[0][1], reward_generator)
        self.action_space = action_space
        self.ob_generators = ob_generator  # a list of ob_generators for each intersection ordered by its int_id
        self.ob_length = ob_generator[0][1].ob_length  # the observation length for each intersection
        self.args = args

        self.graph_setting = graph_setting
        self.neighbor_id = self.graph_setting["NEIGHBOR_ID"]  # neighbor node of node
        self.degree_num = self.graph_setting["NODE_DEGREE_NODE"]  # degree of each intersection
        self.edge_idx = torch.flip(torch.tensor(self.graph_setting["EDGE_IDX"].T, dtype=torch.long), dims=[0])
        self.graph_setting.update({'action_space': self.action_space})
        self.passer = self.graph_setting

        self.direction_dic = {"0": [1, 0, 0, 0], "1": [0, 1, 0, 0], "2": [0, 0, 1, 0],
                              "3": [0, 0, 0, 1]}  # TODO: should refine it

        self.dic_traffic_env_conf = traffic_env_conf
        self.num_agents = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_edges = self.dic_traffic_env_conf["NUM_ROADS"]

        self.vehicle_max = args.vehicle_max
        self.mask_type = args.mask_type
        self.get_attention = args.get_attention

        # DQN setting
        self.batch_size = args.batch_size  # 32
        self.learning_rate = args.learning_rate  # 0.0001
        self.replay_buffer = deque(maxlen=args.replay_buffer_size)
        self.learning_start = args.learning_start  # 2000
        self.update_model_freq = 1
        self.update_target_model_freq = args.update_target_model_freq  # 20
        self.gamma = 0.95  # discount rate
        self.epsilon = args.epsilon  # exploration rate
        self.epsilon_min = args.min_epsilon  # default 0.01
        self.epsilon_decay = args.epsilon_decay  # 0.995
        self.grad_clip = args.grad_clip

        self.world = world
        self.world.subscribe("pressure")
        self.world.subscribe("lane_count")
        self.world.subscribe("lane_waiting_count")

        self.criterion = nn.MSELoss(reduction='mean')
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, alpha=0.9, centered=False, eps=1e-7)
        #for i in self.model.named_parameters():
        #    print(i)

    def _build_model(self):
        """
        layer definition
        """
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        # In: [batch,agent,feature]
        # In: [batch,agent,neighbors,agents]
        # In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        # In.append(Input(shape=(self.num_agents,self.num_neighbors,self.num_agents),name="adjacency_matrix"))
        #TODO: keep no phase
        model = ColightNet(self.ob_length, **self.passer)
        #model = ColightNet(self.action_space.n + self.ob_length, **self.passer)
        # if self.graph_setting["N_LAYERS"]>1:
        #     att_record_all_layers=Concatenate(axis=1)(att_record_all_layers)
        # else:
        #     att_record_all_layers=att_record_all_layers[0]

        # att_record_all_layers=Reshape((self.graph_setting["N_LAYERS"],self.num_agents,self.graph_setting["NUM_HEADS"][self.graph_setting["N_LAYERS"]-1],self.graph_setting["NEIGHBOR_NUM"]+1))(att_record_all_layers)

        # out = Dense(self.action_space.n,kernel_initializer='random_normal',name='action_layer')(h)
        # out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        # model=Model(inputs=In,outputs=[out,att_record_all_layers])
        print(model)
        return model

    def get_action(self, phase, ob, test_phase=False):
        """
        phase : [agents]
        obs : [agents, ob_length]
        edge_obs : [agents, edge_ob_length]
        return: [batch,agents]
        we should expand the input first to adapt the [batch,] standard
        """
        if not test_phase:
            # print("train_phase")
            if np.random.rand() <= self.epsilon:
                return self.sample(s_size=self.num_agents)
        # ob = self._reshape_ob(ob)
        # act_values = self.model.predict([phase, ob])

        e_ob = torch.tensor(ob, dtype=torch.float32)
        edge = self.edge_idx
        dt = Data(x=e_ob, edge_index=edge)

        #e_phase = F.one_hot(torch.tensor(phase, dtype=torch.long), self.action_space.n)
        #x = torch.concat([e_ob, e_phase], dim=1)
        #data = Data(x=x, edge_index=self.graph_setting['edge_idx'])

        # observations = np.concatenate([phase,ob],axis=-1)
        # observations = observations[np.newaxis,:]
        if self.get_attention:
            # TODO: no phase here
            actions = self.model.forward(x=dt.x, edge_index=dt.edge_index)
            att = self.get_attention
            #TODO: implement att
            actions = actions.detach().numpy()
            return np.argmax(actions, axis=1)  # [batch, agents],[batch,agent,nv,neighbor]
        else:
            actions = self.model.forward(x=dt.x, edge_index=dt.edge_index)
            actions = actions.detach().numpy()
            return np.argmax(actions, axis=1)  # batch, agents

    def sample(self, s_size):
        return np.random.randint(0, self.action_space.n, s_size)
        # return self.action_space.sample()

    def get_reward(self, reward_type="vehicleNums"):
        # reward_type should in ["pressure", "vehicleNums"]
        # The colight use queue length on the approaching lane l at time t (minus)
        # lane_nums = self.world.get_info("")
        # should care about the order of world.intersection
        if reward_type == "pressure":
            pressures = self.world.get_info("pressure")  # we should change the dict pressure to list
            pressure_reward = []  # order by the int_id of agents
            for i in range(self.num_agents):
                node_id = self.graph_setting["ID2INTER_MAPPING"][i]
                pressure_reward.append(-pressures[node_id])
            pressure_reward = np.array(pressure_reward)
            return pressure_reward
        elif reward_type == "vehicleNums":
            vehicle_reward = []
            vehicle_nums = self.world.get_info("lane_waiting_count")
            for i in range(self.num_agents):
                node_id = self.graph_setting["ID2INTER_MAPPING"][i]
                node_dict = self.world.id2intersection[node_id]
                nvehicles = 0
                tmp_vehicle = []
                in_lanes = []
                for road in node_dict.in_roads:
                    from_zero = (road["startIntersection"] == node_dict.id) if self.world.RIGHT else (
                                road["endIntersection"] == i.id)
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
        else:
            raise KeyError("reward_type should in ['pressure', 'vehicleNums']")

    def get_ob(self):
        """
        return: obersavtion of node, observation of edge
        """
        x_obs = []  # num_agents * lane_nums,
        for i in range(len(self.ob_generators)):
            x_obs.append((self.ob_generators[i][1].generate()) / self.vehicle_max)
        # construct edge infomation
        x_obs = np.array(x_obs)
        return x_obs

    def remember(self, ob, phase, action, reward, next_ob, next_phase):
        self.replay_buffer.append((ob, phase, action, reward, next_ob, next_phase))
    """
    def _encode_sample(self, minibatch):
        batch_list = []
        batch_list_p = []
        #ob_t, phase_t, actions_t, rewards_t, ob_tp1, phase_tp1 = list(zip(*minibatch))
        #rewards = torch.tensor(rewards_t, dtype=torch.float32)
        #actions = actions_t
        actions = []
        rewards = []
        for dp in minibatch:
            cat = F.one_hot(torch.tensor(dp[1], dtype=torch.long), self.action_space.n)
            state = torch.tensor(dp[0], dtype=torch.float32)
            x = torch.concat([state, cat], dim=1)
            batch_list.append(Data(x=x, edge_index=self.edge_idx))

            cat_p = F.one_hot(torch.tensor(dp[5], dtype=torch.long), self.action_space.n)
            state_p = torch.tensor(dp[4], dtype=torch.float32)
            x_p = torch.concat([state_p, cat_p], dim=1)
            batch_list_p.append(Data(x=x_p, edge_index=self.edge_idx))
            rewards.append(dp[3])
            actions.append(dp[2])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        actions = actions.view(actions.shape[0] * actions.shape[1])
        return batch_t, batch_tp, rewards, actions
    """

    def _encode_sample(self, minibatch):
        batch_list = []
        batch_list_p = []
        #ob_t, phase_t, actions_t, rewards_t, ob_tp1, phase_tp1 = list(zip(*minibatch))
        #rewards = torch.tensor(rewards_t, dtype=torch.float32)
        #actions = actions_t
        actions = []
        rewards = []
        for dp in minibatch:
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
        actions = actions.view(actions.shape[0] * actions.shape[1])
        return batch_t, batch_tp, rewards, actions

    def replay(self):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._encode_sample(minibatch)
        out = self.target_model.forward(x=b_tp.x, edge_index=b_tp.edge_index, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model.forward(x=b_t.x, edge_index=b_t.edge_index, train=False)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model.forward(x=b_t.x, edge_index=b_t.edge_index, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        """
        thr = 0
        for i in self.eval_model.named_parameters():
            thr += 1
            if thr == 8:
                print(i[1].grad)
                break
        """
        self.optimizer.step()
        #print('======================after')
        weights = self.model.state_dict()
        for i in self.model.state_dict():
            #print(weights[i].data)
            break
        # print(history.history['loss'])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        #print('=============================')
        weights_b = self.target_model.state_dict()
        for i in self.target_model.state_dict():
            #print('before', '\n')
            #print(weights_b[i].data)
            break

        self.target_model.load_state_dict(weights)
        for i in self.model.state_dict():
            #print('refer', '\n')
            #print(weights[i].data)
            break
        weights_a = self.target_model.state_dict()
        for i in self.target_model.state_dict():
            #print('after', '\n')
            #print(weights_a[i].data)
            break

    def load_model(self, mdir="model/colight_torch", prefix='', e=0):
        """
        mdir is the path of model
        """
        # name = "netlight_agent_{}".format(self.iid)
        # model_name = os.path.join(mdir, name)
        name = "colight_agent_{}_{}.pt".format(prefix, e)
        model_name = os.path.join(mdir, name)
        self.model = ColightNet(self.ob_length, **self.passer)
        self.model.load_state_dict(torch.load(model_name))
        # self.model.load_weights(model_name)

    def save_model(self, mdir="model/colight_torch", prefix='', e=0):
        name = "colight_agent_{}_{}.pt".format(prefix, e)
        model_name = os.path.join(mdir, name)
        torch.save(self.model.state_dict(), model_name)
        # self.model.save_weights(model_name)