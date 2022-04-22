from . import RLAgent
from common.registry import Registry
from agent import utils
import numpy as np
import os
import random
from collections import deque
import gym

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
from generator.intersection_vehicle import IntersectionVehicleGenerator
from agent.utils import GetDataSet
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict


@Registry.register_model('intellilight')
class IntelliLightAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world)
        self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1
        self.rank = rank
        self.phase_list = [i for i in range(self.action_space.n)]
        self.action_size = len(self.phase_list)

        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        # TODO make sure params average=None or all or road(origin is lane).
        self.ob_generator = [
            LaneVehicleGenerator(
                self.world, inter_obj, ["lane_waiting_count"], in_only=True, average="lane"),
            LaneVehicleGenerator(
                self.world, inter_obj, ["lane_count"], in_only=True, average="lane"),
            LaneVehicleGenerator(
                self.world, inter_obj, ["lane_waiting_time_count"], in_only=True, average="lane"),
            IntersectionVehicleGenerator(
                self.world, inter_obj, targets=["vehicle_map"])
        ]
        self.reward_generator = [
            LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count", "lane_delay", "lane_waiting_time_count"],
                                 in_only=True, average="all"),
            IntersectionVehicleGenerator(
                self.world, inter_obj, targets=["passed_count", "passed_time_count"])
        ]

        self.gamma = Registry.mapping['model_mapping']['model_setting'].param['gamma']
        self.gamma_pretrain = Registry.mapping['model_mapping']['model_setting'].param['gamma_pretrain']
        self.epsilon = Registry.mapping['model_mapping']['model_setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['model_setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['model_setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['model_setting'].param['learning_rate']
        self.batch_size = Registry.mapping['model_mapping']['model_setting'].param['batch_size']
        self.update_peroid = Registry.mapping['model_mapping']['model_setting'].param['update_peroid']
        self.update_q_bar_freq = Registry.mapping['trainer_mapping']['trainer_setting'].param['update_peroid']
        self.sample_size = Registry.mapping['model_mapping']['model_setting'].param['sample_size']
        self.sample_size_pretrain = Registry.mapping['model_mapping']['model_setting'].param['sample_size_pretrain']
        self.separate_memory = Registry.mapping['model_mapping']['model_setting'].param['separate_memory']
        self.priority_sampling = Registry.mapping['model_mapping']['model_setting'].param['priority_sampling']
        self.patience = Registry.mapping['model_mapping']['model_setting'].param['patience']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['traffic_setting']
        
        self.last_phase = self.phase_list[0]
        self.num_phases = len(self.phase_list)
        self.state = None
        self.action = None
        self.average_reward = None
        self.num_actions = 2

        self.q_network = QNet(self.num_phases, self.num_actions)
        self.q_network_bar = QNet(self.num_phases, self.num_actions)
        self.save_model("init_model_torch")

        self.update_outdated = 0
        self.q_bar_outdated = 0

        if not Registry.mapping['model_mapping']['model_setting'].param['separate_memory']:
            self.memory = self.build_memory()
        else:
            self.memory = self.build_memory_separate()

        self.last_action = self.phase_list[0]
        self.action = self.phase_list[-1]

        self.optimizer = torch.optim.RMSprop(self.q_network.parameters(
        ), lr=self.learning_rate, alpha=0.9, centered=False, eps=1e-7)
        self.loss_func = nn.MSELoss()
        self.init_network_bar()

    # TODO check whether to implemete with State,or directly concate them in model.
    # TODO check dims
    def get_ob(self):
        obs_lane = [self.ob_generator[i].generate() for i in range(3)]
        obs_intersection = self.ob_generator[3].generate()

        lane_queue = obs_lane[0]
        lane_num_vehicles = obs_lane[1]
        lane_waiting_time = obs_lane[2]
        map_of_vehicles = obs_intersection[0]
        cur_phase = self.world.intersections[self.idx].current_phase
        next_phase = self.next_phase(cur_phase)
        time_this_phase = self.world.intersections[self.idx].current_phase_time
        state = State(
            queue_length=np.reshape(np.array(lane_queue), newshape=(1, 8)),
            num_of_vehicles=np.reshape(
                np.array(lane_num_vehicles), newshape=(1, 8)),
            waiting_time=np.reshape(
                np.array(lane_waiting_time), newshape=(1, 8)),
            map_feature=np.reshape(
                np.array(map_of_vehicles), newshape=(1, 1, 150, 150)),
            cur_phase=np.reshape(np.array([cur_phase]), newshape=(1, 1)),
            next_phase=np.reshape(np.array([next_phase]),
                                  newshape=(1, 1)),
            time_this_phase=np.reshape(
                np.array([time_this_phase]), newshape=(1, 1)),
            if_terminal=False
        )
        return state

    # TODO check dims
    def get_reward(self):
        reward_lane = self.reward_generator[0].generate()
        reward_intersection = self.reward_generator[1].generate()

        reward_components = {}

        reward_components["queue_length"] = reward_lane[0]
        reward_components["delay"] = reward_lane[1]
        reward_components["waiting_time"] = reward_lane[2]

        reward_components["c"] = 1 - (self.last_action == self.action)

        reward_components["passed_count"] = reward_intersection[0]
        reward_components["passed_time_count"] = reward_intersection[1]

        reward = 0
        weights = Registry.mapping['world_mapping']['traffic_setting'].param['REWARD_WEIGHTS']
        components = Registry.mapping['world_mapping']['traffic_setting'].param['REWARD_COMPONENTS']
        for i in range(len(components)):
            reward += weights[i] * reward_components[components[i]]
        return reward
    
    # TODO check dims
    def get_action(self, ob, train=False):
        """choose the best action"""
        self.q_network.eval()
        self.last_action = self.action
        state_ = self.to_tensor(self.convert_state_to_input(ob))
        q_values = self.q_network(state_, train)
        keep_or_change = torch.argmax(q_values[0])

        self.action = self.next_phase(
            self.action) if keep_or_change else self.action

        return self.action  # action for env(dim=8)
    
    # TODO check dims
    def next_phase(self, phase):
        next_phase = (phase + 1) % self.action_space.n
        return next_phase

    def choose(self, state, count, if_pretrain):
        ''' choose action based on epsilon-greedy '''
        state_ = self.to_tensor(self.convert_state_to_input(state))
        q_values = self.q_network(state_)

        # MemoryStore:0变换，1保持
        if if_pretrain:
            keep_or_change = torch.argmax(q_values[0])
        else:
            # continue explore new Random Action
            if random.random() <= self.epsilon:
                keep_or_change = random.randrange(len(q_values[0]))
                # print("##Explore")
            else:  # exploitation
                keep_or_change = torch.argmax(q_values[0]).item()
            # 当策略到达一定程度时，将探索率减小，开发率提高
            if self.epsilon > self.epsilon_min and count >= 2000:
                self.epsilon *= self.epsilon_decay

        self.action = self.next_phase(
            self.action) if keep_or_change else self.action
        return self.action, q_values, self.epsilon

    def set_update_outdated(self):

        self.update_outdated = - 2 * self.update_peroid
        self.q_bar_outdated = 2 * self.update_q_bar_freq

    def reset_update_count(self):

        self.update_outdated = 0
        self.q_bar_outdated = 0

    def load_model(self, file_name):
        self.q_network.load_state_dict(torch.load(os.path.join(
            paras["PATH_TO_MODEL"], "%s_q_network_torch.pt" % file_name)))
        # self.q_network = load_model(os.path.join(
        #     paras["PATH_TO_MODEL"], "%s_q_network.h5" % file_name), custom_objects={'Selector': Selector})

    def save_model(self, file_name):
        if not os.path.exists(paras["PATH_TO_MODEL"]):
            os.makedirs(paras["PATH_TO_MODEL"])
        torch.save(self.q_network.state_dict(), os.path.join(
            paras["PATH_TO_MODEL"], "%s_q_network_torch.pt" % file_name))

    def _sample_memory(self, gamma, with_priority, memory, if_pretrain):

        len_memory = len(memory)

        if not if_pretrain:
            sample_size = min(self.sample_size, len_memory)
        else:
            sample_size = min(self.sample_size_pretrain, len_memory)

        if with_priority:
            # sample with priority
            sample_weight = []
            for i in range(len_memory):
                state, action, reward, next_state = memory[i]

                if state.if_terminal:
                    next_estimated_reward = 0
                else:

                    next_estimated_reward = self._get_next_estimated_reward(
                        next_state)

                total_reward = reward + gamma * next_estimated_reward
                target = self.predict(self.convert_state_to_input(state))
                pre_target = np.copy(target)
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                sample_weight.append(weight)

            priority = self._cal_priority(sample_weight)
            p = random.choices(range(len(priority)),
                               weights=priority, k=sample_size)
            sampled_memory = np.array(memory)[p]
        else:
            sampled_memory = random.sample(memory, sample_size)

        return sampled_memory

    def update_network(self, if_pretrain, use_average, current_time):
        ''' update Q network '''

        if current_time - self.update_outdated < self.update_peroid:
            return

        self.update_outdated = current_time

        # prepare the samples
        if if_pretrain:
            gamma = self.gamma_pretrain
        else:
            gamma = self.gamma

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf.param["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        # get average state-action reward
        if :
            self.average_reward = self._cal_average_separate(self.memory)
        else:
            self.average_reward = self._cal_average(self.memory)

        # ================ sample memory ====================
        if self.separate_memory:
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):

                    sampled_memory = self._sample_memory(
                        gamma=gamma,
                        with_priority=self.priority_sampling,
                        memory=self.memory[phase_i][action_i],
                        if_pretrain=if_pretrain)
                    dic_state_feature_arrays, Y = self.get_sample(
                        sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        else:
            sampled_memory = self._sample_memory(
                gamma=gamma,
                with_priority=self.priority_sampling,
                memory=self.memory,
                if_pretrain=if_pretrain)
            dic_state_feature_arrays, Y = self.get_sample(
                sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        # ================ sample memory ====================
        Xs = [np.array(dic_state_feature_arrays[feature_name])
              for feature_name in self.dic_traffic_env_conf.param["LIST_STATE_FEATURE"]]

        Y = np.array(Y)
        sample_weight = np.ones(len(Y))
        # shuffle the training samples, especially for different phases and actions
        Xs, Y, _ = self._unison_shuffled_copies(Xs, Y, sample_weight)

        # ============================  training  =======================================

        self.train_network(Xs, Y, current_time, if_pretrain)
        self.q_bar_outdated += 1
        self.forget(if_pretrain=if_pretrain)
        print("done")

    def update_network_bar(self):
        ''' update Q bar '''
        if self.q_bar_outdated >= self.update_q_bar_freq:
            self.q_network_bar.load_state_dict(self.q_network.state_dict())

    def init_network_bar(self):
        weights = self.q_network.state_dict()
        self.q_network_bar.load_state_dict(weights)

    def train_network(self, Xs, Y, prefix, if_pretrain):
        if if_pretrain:
            epochs = Registry.mapping['trainer_mapping']['trainer_setting'].param["EPOCHS_PRETRAIN"]
        else:
            epochs = Registry.mapping['trainer_mapping']['trainer_setting'].param["EPOCHS"]

        # batch_size: min(20,300)
        batch_size = min(self.batch_size, len(Y))
        states = []
        '''
        order:
        queue_length,num_of_vehicles,
        waiting_time,map_feature, 
        cur_phase, next_phase
        '''
        for idx_batch in range(len(Y)):
            Y[idx_batch] = torch.from_numpy(Y[idx_batch]).float()
            out = []
            for idx_state in range(len(Xs)):
                out.append(torch.from_numpy(Xs[idx_state][idx_batch]).float())
            states.append(out)
        # 将array转换为tensor
        dataset = GetDataSet(states, Y)

        for epoch in range(epochs):
            # len(data) = 15(300/20)
            self.q_network.train()
            data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, (x, y) in enumerate(data):
                y_pred = self.q_network(x)
                loss = self.loss_func(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("Epoch: {}/{}, Step: {}/{} Loss: {:.4f}"
                #       .format(epoch+1, epochs, i+1, batch_size, loss.item()))
        self.save_model(prefix)

    def _cal_average_separate(self, sample_memory):
        ''' Calculate average rewards for different cases '''

        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                len_sample_memory = len(sample_memory[phase_i][action_i])
                if len_sample_memory > 0:
                    list_reward = []
                    for i in range(len_sample_memory):
                        state, action, reward, _ = sample_memory[phase_i][action_i][i]
                        list_reward.append(reward)
                    average_reward[phase_i][action_i] = np.average(list_reward)
        return average_reward

    def get_sample(self, memory_slice, dic_state_feature_arrays, Y, gamma, prefix, use_average):
        len_memory_slice = len(memory_slice)

        f_samples = open(os.path.join(
            paras["PATH_TO_OUTPUT"], "{0}_memory".format(prefix)), "a")

        for i in range(len_memory_slice):
            state, action, reward, next_state = memory_slice[i]
            for feature_name in self.dic_traffic_env_conf.param["LIST_STATE_FEATURE"]:
                # 因为是单智能体，所以后面要加[0]
                dic_state_feature_arrays[feature_name].append(
                    getattr(state, feature_name)[0])

            if state.if_terminal:
                next_estimated_reward = 0
            else:
                next_estimated_reward = self._get_next_estimated_reward(
                    next_state)
            total_reward = reward + gamma * \
                np.array(next_estimated_reward, copy=False)

            if not use_average:
                state_ = self.to_tensor(self.convert_state_to_input(state))
                target = np.array(self.q_network(state_, train=False))
            else:
                target = np.copy(
                    np.array([self.average_reward[state.cur_phase[0][0]]]))

            pre_target = np.copy(target)
            target[0][action] = np.array(total_reward)
            Y.append(target[0])

            for feature_name in self.dic_traffic_env_conf.param["LIST_STATE_FEATURE"]:
                if "map" not in feature_name:
                    f_samples.write("{0}\t".format(
                        str(getattr(state, feature_name))))
            f_samples.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                str(pre_target), str(target),
                str(action), str(reward), str(next_estimated_reward)
            ))
        f_samples.close()

        return dic_state_feature_arrays, Y

    def _get_next_estimated_reward(self, next_state):

        next_state_ = self.convert_state_to_input(next_state)
        next_state_ = self.to_tensor(next_state_)
        if self.dic_traffic_env_conf.param["DDQN"]:
            a_max = torch.argmax(self.q_network(next_state_)[0])
            next_estimated_reward = self.q_network_bar(
                next_state_, train=False)[0][a_max]
            return next_estimated_reward
        else:
            next_estimated_reward = torch.max(
                self.q_network_bar(next_state_, train=False), dim=1)[0]
            return next_estimated_reward

    def convert_state_to_input(self, state):
        return [getattr(state, feature_name)
                for feature_name in self.dic_traffic_env_conf.param["LIST_STATE_FEATURE"]]

    def to_tensor(self, state):
        for i in range(len(state)):
            state[i] = torch.from_numpy(state[i]).float()
        return state

    def build_memory_separate(self):
        memory_list = []
        for i in range(self.num_phases):
            memory_list.append([[] for j in range(self.num_actions)])
        return memory_list

    def remember(self, state, action, reward, next_state):
        if self.separate_memory:
            ''' log the history separately '''

            self.memory[state.cur_phase[0][0]][action].append(
                [state, action, reward, next_state])
        else:
            self.memory.append([state, action, reward, next_state])

    def forget(self, if_pretrain):

        if self.separate_memory:
            ''' remove the old history if the memory is too large, in a separate way '''
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    if if_pretrain:
                        random.shuffle(self.memory[phase_i][action_i])
                    if len(self.memory[phase_i][action_i]) > self.buffer_size:
                        print("length of memory (state {0}, action {1}): {2}, before forget".format(
                            phase_i, action_i, len(self.memory[phase_i][action_i])))
                        self.memory[phase_i][action_i] = self.memory[phase_i][action_i][-self.buffer_size:]
                    print("length of memory (state {0}, action {1}): {2}, after forget".format(
                        phase_i, action_i, len(self.memory[phase_i][action_i])))
        else:
            if len(self.memory) > self.buffer_size:
                print("length of memory: {0}, before forget".format(
                    len(self.memory)))
                self.memory = self.memory[-self.buffer_size:]
            print("length of memory: {0}, after forget".format(
                len(self.memory)))

    def _cal_average(self, sample_memory):

        list_reward = []
        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            list_reward.append([])
            for action_i in range(self.num_actions):
                list_reward[phase_i].append([])
        for [state, action, reward, _] in sample_memory:
            phase = state.cur_phase[0][0]
            list_reward[phase][action].append(reward)

        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                if len(list_reward[phase_i][action_i]) != 0:
                    average_reward[phase_i][action_i] = np.average(
                        list_reward[phase_i][action_i])

        return average_reward

    @staticmethod
    # referenced by sample_memory()
    def _cal_priority(sample_weight):
        pos_constant = 0.0001
        alpha = 1
        sample_weight_np = np.array(sample_weight)
        sample_weight_np = np.power(
            sample_weight_np + pos_constant, alpha) / sample_weight_np.sum()
        return sample_weight_np

    @staticmethod
    # referenced by update network()
    def _unison_shuffled_copies(Xs, Y, sample_weight):
        p = np.random.permutation(len(Y))
        new_Xs = []
        for x in Xs:
            assert len(x) == len(Y)
            new_Xs.append(x[p])
        return new_Xs, Y[p], sample_weight[p]

    @staticmethod
    def _unison_shuffle(Xs, Y):
        p = torch.randperm(len(Y))
        new_Xs = []
        for _ in Xs:
            new_Xs.append(Xs[p])
        return new_Xs, Y[p]


class QNet(nn.Module):
    def __init__(self, num_phases, num_actions):
        super(QNet, self).__init__()

        self.cnn = CNN()
        # shared dense layer
        self.share = nn.Linear(1*426, Registry.mapping['model_mapping']['model_setting']["d_dense"])
        self.num_phases = num_phases
        self.num_actions = num_actions
        self.phasenet = []
        # build 8 phase selector layer
        for phase in range(self.num_phases):
            self.phasenet.append(
                PhaseNet(Registry.mapping['model_mapping']['model_setting']["d_dense"], self.num_actions, phase))

    def _forward(self, states, train):

        # # add cnn to image features
        list_all_flatten_feature = []
        for i in range(len(states)):
            # i=3 is map_features
            if states[i].dim() > 2:
                if train:
                    self.cnn.train()
                    list_all_flatten_feature.append(self.cnn(states[i]))
                else:
                    self.cnn.eval()
                    list_all_flatten_feature.append(self.cnn(states[i]))
            else:
                list_all_flatten_feature.append(states[i])

        cur_phase = states[5]

        all_flatten_feature = torch.cat(tuple(list_all_flatten_feature), dim=1)

        shareout = self.share(all_flatten_feature)
        q_values = torch.zeros(size=(1, self.num_actions), dtype=torch.float32)

        # batch_size = len(states[0])
        # q_values = torch.zeros(size=(batch_size,self.num_actions),dtype=torch.float32)
        for phase in range(self.num_phases):
            out = self.phasenet[phase](shareout, cur_phase)
            q_values = torch.add(q_values, out)
        return q_values

    def forward(self, states, train=True):
        if train:
            return self._forward(states, train)
        else:
            with torch.no_grad():
                return self._forward(states, train)


class PhaseNet(nn.Module):
    def __init__(self, in_states, num_actions, select):
        super(PhaseNet, self).__init__()

        self.fc1 = nn.Linear(in_states, Registry.mapping['model_mapping']['model_setting']["d_dense"])
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(Registry.mapping['model_mapping']['model_setting']["d_dense"], num_actions)
        self.selector = Selector(select)

    def forward(self, in_features, cur_phase):
        hidden1 = self.fc1(in_features)
        hidden1 = self.act(hidden1)
        hidden2 = self.fc2(hidden1)
        selectout = self.selector(cur_phase)
        out = selectout*hidden2
        return out


class Selector(nn.Module):
    def __init__(self, select):
        super(Selector, self).__init__()
        self.select = np.array([[select]])
        self.select_neuron = torch.tensor(self.select, dtype=torch.float32)

    def forward(self, x):
        return torch.eq(self.select_neuron, x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            OrderedDict([
                (
                    "conv_1",
                    nn.Conv2d(
                        in_channels=1,  # 输入图片的维度
                        out_channels=32,  # 输出图片的维度
                        kernel_size=8,  # 8x8的卷积核，相当于过滤器
                        stride=4,  # 卷积核在图上滑动，每隔一个扫一次
                        padding=3,  # 给图外边补上0
                        bias=False,
                    )),
                ("bn_1", nn.BatchNorm2d(32)),
                ("relu_1", nn.ReLU()),
                ("maxpooling_1", nn.MaxPool2d(kernel_size=2)),
                ("dropout_1", nn.Dropout2d(0.3))
            ]))
        # 第二层卷积
        self.layer2 = nn.Sequential(
            OrderedDict([
                (
                    "conv_2",
                    nn.Conv2d(
                        in_channels=32,  
                        out_channels=16,  
                        kernel_size=3,  
                        stride=2, 
                        padding=1, 
                        bias=False,
                    )),
                ("bn_2", nn.BatchNorm2d(16)),
                ("relu_2", nn.ReLU()),
                ("maxpooling_2", nn.MaxPool2d(kernel_size=2)),
                ("dropout_2", nn.Dropout2d(0.3))
            ]))

        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        return out


class State(object):
    # ==========================

    D_QUEUE_LENGTH = (8, )
    D_NUM_OF_VEHICLES = (8, )
    D_WAITING_TIME = (8, )
    D_MAP_FEATURE = (
        1,
        150,
        150,
    )
    D_CUR_PHASE = (1, )
    D_NEXT_PHASE = (1, )
    D_TIME_THIS_PHASE = (1, )

    D_IF_TERMINAL = (1, )
    D_HISTORICAL_TRAFFIC = (6, )

    # ==========================

    def __init__(self, queue_length, num_of_vehicles, waiting_time,
                 map_feature, cur_phase, next_phase, time_this_phase,
                 if_terminal):

        self.queue_length = queue_length
        self.num_of_vehicles = num_of_vehicles
        self.waiting_time = waiting_time
        self.map_feature = map_feature

        self.cur_phase = cur_phase
        self.next_phase = next_phase
        self.time_this_phase = time_this_phase

        self.if_terminal = if_terminal

        self.historical_traffic = None


