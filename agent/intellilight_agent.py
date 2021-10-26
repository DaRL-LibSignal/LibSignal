from . import RLAgent
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.merge import concatenate, add
import random
import os

paras = {"LEARNING_RATE": 0.001,
"UPDATE_PERIOD": 300,
"SAMPLE_SIZE": 300,
"SAMPLE_SIZE_PRETRAIN": 3000,
"BATCH_SIZE": 20,
"EPOCHS": 50,
"EPOCHS_PRETRAIN": 500,
"SEPARATE_MEMORY": True,
"PRIORITY_SAMPLING": False,
"UPDATE_Q_BAR_FREQ": 5,
"GAMMA": 0.8,
"GAMMA_PRETRAIN": 0,
"MAX_MEMORY_LEN": 1000,
"EPSILON": 0.1,
"PATIENCE": 10,
"PHASE_SELECTOR": True,
"DDQN": False,
"D_DENSE": 20,
"RUN_COUNTS": 72000,
"RUN_COUNTS_PRETRAIN": 10000,
"BASE_RATIO": [10, 10, 10, 10, 10, 10, 10, 10],
"PATH_TO_OUTPUT": "./intellilight/memory/",
"PATH_TO_MODEL":"./intellilight/model",
"STATE_FEATURE": {  "queue_length": True,
                    "num_of_vehicles": True,
                    "waiting_time": True,
                    "historical_traffic": False,
                    "map_feature": True,
                    "cur_phase": True,
                    "next_phase": True,
                    "time_this_phase": False,
                    "if_terminal": False
                    },
"LIST_STATE_FEATURE": ["queue_length", "num_of_vehicles", "waiting_time", "map_feature", "cur_phase", "next_phase"],
"REWARD_WEIGHTS": [-0.25, -0.25, -0.25, -5, 1, 1],
"REWARD_COMPONENTS": ["queue_length", "delay", "waiting_time", "c", "passed_count"]#, "passed_time_count"]
}

class IntelliLightAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, world, idx):
        super().__init__(action_space, ob_generator, reward_generator)

        self.idx = idx
        self.world = world

        self.phase_list = [i for i in range(self.action_space.n)]

        self.action_size = len(self.phase_list) # action for env

        self.ob_generator = ob_generator
        self.reward_generator = reward_generator
        self.last_phase = self.phase_list[0]



        self.num_phases = len(self.phase_list)
        self.state = None
        self.action = None
        self.memory = []
        self.average_reward = None

        self.num_actions = 2 # action of intellilight

        self.q_network = self.build_network()
        self.save_model("init_model")
        self.update_outdated = 0

        self.q_network_bar = self.build_network_from_copy(self.q_network)
        self.q_bar_outdated = 0
        if not paras["SEPARATE_MEMORY"]:
            self.memory = self.build_memory()
        else:
            self.memory = self.build_memory_separate()
        self.average_reward = None



        self.last_action = self.phase_list[0]
        self.action = self.phase_list[-1]


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

        # print(obs_lane)
        state = State(
            queue_length=np.reshape(np.array(lane_queue), newshape=(1, 8)),
            num_of_vehicles=np.reshape(np.array(lane_num_vehicles), newshape=(1, 8)),
            waiting_time=np.reshape(np.array(lane_waiting_time), newshape=(1, 8)),
            map_feature=np.reshape(np.array(map_of_vehicles), newshape=(1, 150, 150, 1)),
            cur_phase=np.reshape(np.array([cur_phase]), newshape=(1, 1)),
            next_phase=np.reshape(np.array([next_phase]),
                                  newshape=(1, 1)),
            time_this_phase=np.reshape(np.array([time_this_phase]), newshape=(1, 1)),
            if_terminal=False
        )

        # print("----State----")
        # print(state.queue_length)
        # print(state.num_of_vehicles)
        # print(state.waiting_time)
        # print(state.map_feature.shape)
        # for i in state.map_feature[0]:
        #     for j in i:
        #         print(j[0], end=" ")
        #     print("\n")
        # print(state.cur_phase)
        # print(state.next_phase)
        # print(state.time_this_phase)


        return state

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

        # print("----Reward----")
        # for i in reward_components:
        #     print(i, reward_components[i])
        #
        reward = 0
        weights = paras["REWARD_WEIGHTS"]
        components = paras["REWARD_COMPONENTS"]
        for i in range(len(components)):
            reward += weights[i] * reward_components[components[i]]

        return reward
    #
    def get_action(self, ob):
        """choose the best action"""
        self.last_action = self.action

        state_ = self.convert_state_to_input(ob)
        q_values = self.q_network.predict(state_)
        keep_or_change = np.argmax(q_values[0])

        self.action = self.next_phase(self.action) if keep_or_change else self.action

        return self.action # action for env(dim=8)

    def next_phase(self, phase):
        next_phase = (phase + 1) % self.action_space.n
        return next_phase

    def choose(self, state, count, if_pretrain):
        ''' choose action based on epsilon-greedy '''
        state_ = self.convert_state_to_input(state)

        q_values = self.q_network.predict(state_)
        if if_pretrain:
            keep_or_change = np.argmax(q_values[0])
        else:
            if random.random() <= paras["EPSILON"]:  # continue explore new Random Action
                keep_or_change = random.randrange(len(q_values[0]))
                # print("##Explore")
            else:  # exploitation
                keep_or_change = np.argmax(q_values[0])
            if paras["EPSILON"] > 0.001 and count >= 20000:
                paras["EPSILON"] = paras["EPSILON"] * 0.9999

        self.action = self.next_phase(self.action) if keep_or_change else self.action
        return self.action, q_values

    def set_update_outdated(self):

        self.update_outdated = - 2 * paras["UPDATE_PERIOD"]
        self.q_bar_outdated = 2 * paras["UPDATE_Q_BAR_FREQ"]

    def reset_update_count(self):

        self.update_outdated = 0
        self.q_bar_outdated = 0

    def convert_state_to_input(self, state):

        ''' convert a state struct to the format for neural network input'''

        return [getattr(state, feature_name)
                for feature_name in paras["LIST_STATE_FEATURE"]]

    def build_network(self):

        '''Initialize a Q network'''

        # initialize feature node
        dic_input_node = {}
        for feature_name in paras["LIST_STATE_FEATURE"]:
            dic_input_node[feature_name] = Input(shape=getattr(State, "D_"+feature_name.upper()),
                                                     name="input_"+feature_name)

        # add cnn to image features
        dic_flatten_node = {}
        for feature_name in paras["LIST_STATE_FEATURE"]:
            if len(getattr(State, "D_"+feature_name.upper())) > 1:
                dic_flatten_node[feature_name] = self._cnn_network_structure(dic_input_node[feature_name])
            else:
                dic_flatten_node[feature_name] = dic_input_node[feature_name]

        # concatenate features
        list_all_flatten_feature = []
        for feature_name in paras["LIST_STATE_FEATURE"]:
            list_all_flatten_feature.append(dic_flatten_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        # shared dense layer
        shared_dense = self._shared_network_structure(all_flatten_feature, paras["D_DENSE"])

        # build phase selector layer
        if "cur_phase" in paras["LIST_STATE_FEATURE"] and paras["PHASE_SELECTOR"]:
            list_selected_q_values = []
            for phase in range(self.num_phases):
                locals()["q_values_{0}".format(phase)] = self._separate_network_structure(
                    shared_dense, paras["D_DENSE"], self.num_actions, memo=phase)
                locals()["selector_{0}".format(phase)] = Selector(
                    phase, name="selector_{0}".format(phase))(dic_input_node["cur_phase"])
                locals()["q_values_{0}_selected".format(phase)] = Multiply(name="multiply_{0}".format(phase))(
                    [locals()["q_values_{0}".format(phase)],
                     locals()["selector_{0}".format(phase)]]
                )
                list_selected_q_values.append(locals()["q_values_{0}_selected".format(phase)])
            q_values = Add()(list_selected_q_values)
        else:
            q_values = self._separate_network_structure(shared_dense, paras["D_DENSE"], self.num_actions)

        network = Model(inputs=[dic_input_node[feature_name]
                                for feature_name in paras["LIST_STATE_FEATURE"]],
                        outputs=q_values)
        network.compile(optimizer=RMSprop(lr=paras["LEARNING_RATE"]),
                        loss="mean_squared_error")
        network.summary()

        return network

    def build_memory_separate(self):
        memory_list=[]
        for i in range(self.num_phases):
            memory_list.append([[] for j in range(self.num_actions)])
        return memory_list

    def remember(self, state, action, reward, next_state):
        if paras["SEPARATE_MEMORY"]:
            ''' log the history separately '''
            # print(self.memory)
            # print(state.cur_phase)
            # print(action)
            # print(len(self.memory))
            self.memory[state.cur_phase[0][0]][action].append([state, action, reward, next_state])
        else:
            self.memory.append([state, action, reward, next_state])

    def forget(self, if_pretrain):

        if paras["SEPARATE_MEMORY"]:
            ''' remove the old history if the memory is too large, in a separate way '''
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    if if_pretrain:
                        random.shuffle(self.memory[phase_i][action_i])
                    if len(self.memory[phase_i][action_i]) > paras["MAX_MEMORY_LEN"]:
                        print("length of memory (state {0}, action {1}): {2}, before forget".format(
                            phase_i, action_i, len(self.memory[phase_i][action_i])))
                        self.memory[phase_i][action_i] = self.memory[phase_i][action_i][-paras["MAX_MEMORY_LEN"]:]
                    print("length of memory (state {0}, action {1}): {2}, after forget".format(
                        phase_i, action_i, len(self.memory[phase_i][action_i])))
        else:
            if len(self.memory) > paras["MAX_MEMORY_LEN"]:
                print("length of memory: {0}, before forget".format(len(self.memory)))
                self.memory = self.memory[-paras["MAX_MEMORY_LEN"]:]
            print("length of memory: {0}, after forget".format(len(self.memory)))

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
                    average_reward[phase_i][action_i] = np.average(list_reward[phase_i][action_i])

        return average_reward

    def _cal_average_separate(self,sample_memory):
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
                    average_reward[phase_i][action_i]=np.average(list_reward)
        return average_reward

    def get_sample(self, memory_slice, dic_state_feature_arrays, Y, gamma, prefix, use_average):

        len_memory_slice = len(memory_slice)

        f_samples = open(os.path.join(paras["PATH_TO_OUTPUT"], "{0}_memory".format(prefix)), "a")

        for i in range(len_memory_slice):
            state, action, reward, next_state = memory_slice[i]
            for feature_name in paras["LIST_STATE_FEATURE"]:
                dic_state_feature_arrays[feature_name].append(getattr(state, feature_name)[0])

            if state.if_terminal:
                next_estimated_reward = 0
            else:
                next_estimated_reward = self._get_next_estimated_reward(next_state)
            total_reward = reward + gamma * next_estimated_reward
            if not use_average:
                target = self.q_network.predict(
                    self.convert_state_to_input(state))
            else:
                target = np.copy(np.array([self.average_reward[state.cur_phase[0][0]]]))

            pre_target = np.copy(target)
            target[0][action] = total_reward
            Y.append(target[0])

            for feature_name in paras["LIST_STATE_FEATURE"]:
                if "map" not in feature_name:
                    f_samples.write("{0}\t".format(str(getattr(state, feature_name))))
            f_samples.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                str(pre_target), str(target),
                str(action), str(reward), str(next_estimated_reward)
            ))
        f_samples.close()

        return dic_state_feature_arrays, Y

    def train_network(self, Xs, Y, prefix, if_pretrain):

        if if_pretrain:
            epochs = paras["EPOCHS_PRETRAIN"]
        else:
            epochs = paras["EPOCHS"]
        batch_size = min(paras["BATCH_SIZE"], len(Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=paras["PATIENCE"], verbose=0, mode='min')

        hist = self.q_network.fit(Xs, Y, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=0, validation_split=0.3, callbacks=[early_stopping])
        self.save_model(prefix)

    def update_network(self, if_pretrain, use_average, current_time):

        ''' update Q network '''

        if current_time - self.update_outdated < paras["UPDATE_PERIOD"]:
            return

        self.update_outdated = current_time

        # prepare the samples
        if if_pretrain:
            gamma = paras["GAMMA_PRETRAIN"]
        else:
            gamma = paras["GAMMA"]

        dic_state_feature_arrays = {}
        for feature_name in paras["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        # get average state-action reward
        if paras["SEPARATE_MEMORY"]:
            self.average_reward = self._cal_average_separate(self.memory)
        else:
            self.average_reward = self._cal_average(self.memory)

        # ================ sample memory ====================
        if paras["SEPARATE_MEMORY"]:
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    sampled_memory = self._sample_memory(
                        gamma=gamma,
                        with_priority=paras["PRIORITY_SAMPLING"],
                        memory=self.memory[phase_i][action_i],
                        if_pretrain=if_pretrain)
                    dic_state_feature_arrays, Y = self.get_sample(
                        sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        else:
            sampled_memory = self._sample_memory(
                gamma=gamma,
                with_priority=paras["PRIORITY_SAMPLING"],
                memory=self.memory,
                if_pretrain=if_pretrain)
            dic_state_feature_arrays, Y = self.get_sample(
                sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        # ================ sample memory ====================

        Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in paras["LIST_STATE_FEATURE"]]
        Y = np.array(Y)
        sample_weight = np.ones(len(Y))
        # shuffle the training samples, especially for different phases and actions
        Xs, Y, _ = self._unison_shuffled_copies(Xs, Y, sample_weight)

        # ============================  training  =======================================

        self.train_network(Xs, Y, current_time, if_pretrain)
        self.q_bar_outdated += 1
        self.forget(if_pretrain=if_pretrain)

    def _sample_memory(self, gamma, with_priority, memory, if_pretrain):

        len_memory = len(memory)

        if not if_pretrain:
            sample_size = min(paras["SAMPLE_SIZE"], len_memory)
        else:
            sample_size = min(paras["SAMPLE_SIZE_PRETRAIN"], len_memory)

        if with_priority:
            # sample with priority
            sample_weight = []
            for i in range(len_memory):
                state, action, reward, next_state = memory[i]

                if state.if_terminal:
                    next_estimated_reward = 0
                else:
                    next_estimated_reward = self._get_next_estimated_reward(next_state)

                total_reward = reward + gamma * next_estimated_reward
                target = self.q_network.predict(
                    self.convert_state_to_input(state))
                pre_target = np.copy(target)
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                sample_weight.append(weight)

            priority = self._cal_priority(sample_weight)
            p = random.choices(range(len(priority)), weights=priority, k=sample_size)
            sampled_memory = np.array(memory)[p]
        else:
            sampled_memory = random.sample(memory, sample_size)

        return sampled_memory

    def load_model(self, file_name):
        self.q_network = load_model(os.path.join(paras["PATH_TO_MODEL"], "%s_q_network.h5" % file_name), custom_objects={'Selector': Selector})

    def save_model(self, file_name):
        if not os.path.exists(paras["PATH_TO_MODEL"]):
            os.makedirs(paras["PATH_TO_MODEL"])
        self.q_network.save(os.path.join(paras["PATH_TO_MODEL"], "%s_q_network.h5" % file_name))

    def build_memory(self):

        return []

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''

        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer=RMSprop(lr=paras["LEARNING_RATE"]),
                        loss="mean_squared_error")
        return network

    @staticmethod
    def _cal_priority(sample_weight):
        pos_constant = 0.0001
        alpha = 1
        sample_weight_np = np.array(sample_weight)
        sample_weight_np = np.power(sample_weight_np + pos_constant, alpha) / sample_weight_np.sum()
        return sample_weight_np

    @staticmethod
    def _unison_shuffled_copies(Xs, Y, sample_weight):
        p = np.random.permutation(len(Y))
        new_Xs = []
        for x in Xs:
            assert len(x) == len(Y)
            new_Xs.append(x[p])
        return new_Xs, Y[p], sample_weight[p]

    @staticmethod
    def _cnn_network_structure(img_features):
        conv1 = conv2d_bn(img_features, 1, filters=32, kernel_size=(8, 8), strides=(4, 4))
        conv2 = conv2d_bn(conv1, 2, filters=16, kernel_size=(4, 4), strides=(2, 2))
        img_flatten = Flatten()(conv2)
        return img_flatten

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_separate_branch_{0}_1".format(memo))(
            state_features)
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_branch_{0}".format(memo))(hidden_1)
        return q_values

    def _get_next_estimated_reward(self, next_state):

        if paras["DDQN"]:
            a_max = np.argmax(self.q_network.predict(
                self.convert_state_to_input(next_state))[0])
            next_estimated_reward = self.q_network_bar.predict(
                self.convert_state_to_input(next_state))[0][a_max]
            return next_estimated_reward
        else:
            next_estimated_reward = np.max(self.q_network_bar.predict(
                self.convert_state_to_input(next_state))[0])
            return next_estimated_reward

    def update_network_bar(self):

        ''' update Q bar '''

        if self.q_bar_outdated >= paras["UPDATE_Q_BAR_FREQ"]:
            self.q_network_bar = self.build_network_from_copy(self.q_network)
            self.q_bar_outdated = 0



class State(object):
    # ==========================

    D_QUEUE_LENGTH = (8,)
    D_NUM_OF_VEHICLES = (8,)
    D_WAITING_TIME = (8,)
    D_MAP_FEATURE = (150,150,1,)
    D_CUR_PHASE = (1,)
    D_NEXT_PHASE = (1,)
    D_TIME_THIS_PHASE = (1,)
    D_IF_TERMINAL = (1,)
    D_HISTORICAL_TRAFFIC = (6,)

    # ==========================

    def __init__(self,
                 queue_length, num_of_vehicles, waiting_time, map_feature,
                 cur_phase,
                 next_phase,
                 time_this_phase,
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


class Selector(Layer):

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def conv2d_bn(input_layer, index_layer,
              filters=16,
              kernel_size=(3, 3),
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  name="conv{0}".format(index_layer))(input_layer)
    bn = BatchNormalization(axis=bn_axis, scale=False, name="bn{0}".format(index_layer))(conv)
    act = Activation('relu', name="act{0}".format(index_layer))(bn)
    pooling = MaxPooling2D(pool_size=2)(act)
    x = Dropout(0.3)(pooling)
    return x