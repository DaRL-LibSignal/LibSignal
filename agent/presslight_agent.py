from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.layers.merge import concatenate
import os


class PressLightAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid, is_virtual=False):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid
        self.ob_generator = ob_generator

        self.memory = deque(maxlen=4000)
        self.learning_start = 4000
        self.meta_test_start = 100
        self.meta_test_update_model_freq = 10
        self.meta_test_update_target_model_freq = 200
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 32

        self.list_feature_name = ['num_of_vehicles', 'cur_phase']

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def get_ob(self):

        obs_lane = [self.ob_generator[0].generate()]
        cur_phase = [self.ob_generator[1].generate()]

        # print(obs_lane)
        state = State(
            num_of_vehicles=np.reshape(np.array(obs_lane[0]), newshape=(-1, 24)),
            cur_phase=np.reshape(np.array([cur_phase]), newshape=(-1, 1))
        )

        return state.to_list()

    def choose(self, ob):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])

    def get_action(self, ob):
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])

    def sample(self):
        return self.action_space.sample()

    def _build_model(self):
        '''right now it has same sturctur as dqn model. later introduce k segmentations of input roads'''
        dic_input_node = {}
        for feature_name in self.list_feature_name:
            dic_input_node[feature_name] = Input(shape=State.dims["D_" + feature_name.upper()],
                                                 name="input_" + feature_name)
        # concatenate features
        list_all_flatten_feature = []
        for feature_name in self.list_feature_name:
            list_all_flatten_feature.append(dic_input_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        fc1 = Dense(20, activation='relu')(all_flatten_feature)
        fc2 = Dense(20, activation='relu')(fc1)
        q_values = Dense(self.action_space.n, activation='linear')(fc2)

        model = Model(inputs=[dic_input_node[feature_name]
                              for feature_name in self.list_feature_name],
                      outputs=q_values)

        model.compile(
            loss='mse',
            optimizer=RMSprop(lr=0.001)
        )
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def _encode_sample(self, minibatch):
        obses_t, actions_t, rewards_t, obses_tp1 = list(zip(*minibatch))
        obs = [np.squeeze(obs_i) for obs_i in list(zip(*obses_t))]
        actions = np.array(actions_t, copy=False)
        rewards = np.array(rewards_t, copy=False)
        next_obs = [np.squeeze(obs_i) for obs_i in list(zip(*obses_tp1))]

        return obs, actions, rewards, next_obs

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs = self._encode_sample(minibatch)
        target = rewards + self.gamma * np.amax(self.target_model.predict(next_obs), axis=1)
        target_f = self.model.predict(obs)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        history = self.model.fit(obs, target_f, epochs=1, verbose=0)
        # print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/presslight", e=0):
        name = "presslight_agent_{}_{}.h5".format(self.iid, e)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/presslight", e=0):
        name = "presslight_agent_{}_{}.h5".format(self.iid, e)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)


class State(object):
    # ==========================
    dims = {
        "D_NUM_OF_VEHICLES":  (24,),
        "D_CUR_PHASE":  (1,)

    }

    # ==========================

    def __init__(self, num_of_vehicles, cur_phase):

        self.num_of_vehicles = num_of_vehicles
        self.cur_phase = cur_phase

    def to_list(self):
        results = [self.num_of_vehicles, self.cur_phase]
        return results