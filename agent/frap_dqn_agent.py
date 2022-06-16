from . import RLAgent
import random
import numpy as np
from collections import deque
import os
from keras.layers import Input, Dense, Flatten, Reshape, Layer, Lambda, RepeatVector, Activation, Embedding, Conv2D, BatchNormalization, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers.merge import concatenate, add, dot, maximum, multiply
from keras import backend as K
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Layer
from keras.callbacks import TensorBoard


DIC_AGENT_CONF = {
    "N_LAYER": 2,
    "ROTATION": True,
    "CONFLICT_MATRIX": True,
    "MERGE": "multiply",
    "BATCH_SIZE": 20,
    "D_DENSE": 20,
    "MAX_LEN": 2000,
    "LOSS_FUNCTION": "mean_squared_error",
    "LEARNING_START": 100,
    "LEARNING_RATE": 0.005,
    "UPDATE_MODEL_FREQ": 60,
    "UPDATE_TARGET_MODEL_FREQ": 60,
    "GAMMA": 0.95,
    "EPSILON": 0.2,
    "EPSILON_MIN": 0.01,
    "EPSILON_DECAY": 0.95,
}

DIC_TRAFFIC_ENV_CONF = {
    "N_LEG": 4,
    "DIC_FEATURE_DIM": dict(
        D_CUR_PHASE=(8,),
        D_LANE_NUM_VEHICLE=(8,),
    ),
    "LIST_STATE_FEATURE": [
        "cur_phase",
        "lane_num_vehicle",
    ],
    "PHASE": [
        'NT_ST',
        'WT_ET',
        'NL_SL',
        'WL_EL',
        'NL_NT',
        'SL_ST',
        'WL_WT',
        'EL_ET',
    ],
    "list_lane_order": ["ET", "EL", "ST", "SL", "WT", "WL", "NT", "NL"],
    "phase_expansion": {
        1: [0, 0, 1, 0, 0, 0, 1, 0],
        2: [1, 0, 0, 0, 1, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 1],
        4: [0, 1, 0, 0, 0, 1, 0, 0],
        5: [0, 0, 0, 0, 0, 0, 1, 1],
        6: [0, 0, 1, 1, 0, 0, 0, 0],
        7: [0, 0, 0, 0, 1, 1, 0, 0],
        8: [1, 1, 0, 0, 0, 0, 0, 0]
    },
    "phase_expansion_4_lane": {
        1: [0, 0, 1, 1],
        2: [1, 1, 0, 0],
    },
}

def slice_tensor(x, index):
    x_shape = K.int_shape(x)
    if len(x_shape) == 3:
        return x[:, index, :]
    elif len(x_shape) == 2:
        return Reshape((1, ))(x[:, index])

def competing_network(p, dic_agent_conf, num_actions):
    # competing network
    dense1 = Dense(dic_agent_conf["D_DENSE"], activation="relu", name="dense1")(p)
    dense2 = Dense(dic_agent_conf["D_DENSE"], activation="relu", name="dense2")(dense1)
    q_values = Dense(num_actions, activation="linear", name="q_values")(dense2)
    return q_values

def relation(x, dic_traffic_env_conf):
    relations = []
    for p1 in dic_traffic_env_conf["PHASE"]:
        zeros = [0, 0, 0, 0, 0, 0, 0]
        count = 0
        for p2 in dic_traffic_env_conf["PHASE"]:
            if p1 == p2:
                continue
            m1 = p1.split("_")
            m2 = p2.split("_")
            if len(list(set(m1 + m2))) == 3:
                zeros[count] = 1
            count += 1
        relations.append(zeros)
    relations = np.array(relations).reshape(1, 8, 7)
    batch_size = K.shape(x)[0]
    constant = K.constant(relations)
    constant = K.tile(constant, (batch_size, 1, 1))
    return constant

class FRAP_DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, world, iid):
        super().__init__(action_space, ob_generator, reward_generator)

        self.world = world
        self.iid = iid
        self.dic_agent_conf = DIC_AGENT_CONF
        self.dic_traffic_env_conf = DIC_TRAFFIC_ENV_CONF
        self.ob_length = ob_generator.ob_length

        self.memory = deque(maxlen=self.dic_agent_conf["MAX_LEN"])
        self.learning_start = self.dic_agent_conf["LEARNING_START"]
        self.update_model_freq = self.dic_agent_conf["UPDATE_MODEL_FREQ"]
        self.update_target_model_freq = self.dic_agent_conf["UPDATE_TARGET_MODEL_FREQ"]
        self.gamma = self.dic_agent_conf["GAMMA"]
        self.epsilon = self.dic_agent_conf["EPSILON"]
        self.epsilon_min = self.dic_agent_conf["EPSILON_MIN"]
        self.epsilon_decay = self.dic_agent_conf["EPSILON_DECAY"]
        self.learning_rate = self.dic_agent_conf["LEARNING_RATE"]
        self.batch_size = self.dic_agent_conf["BATCH_SIZE"]

        self.num_phases = len(self.dic_traffic_env_conf["PHASE"])
        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.action = 0
        self.last_action = 0
        self.memory = []
        self.if_test = 0

    def _build_model(self):
        dic_input_node = {}
        feature_shape = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "phase" in feature_name:  # cur_phase
                _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0],)
            else:  # vehicle
                _shape = (self.ob_length,)
            dic_input_node[feature_name] = Input(shape=_shape, name="input_" + feature_name)
            feature_shape[feature_name] = _shape[0]

        p = Activation('sigmoid')(Embedding(2, 4, input_length=feature_shape["cur_phase"])(dic_input_node["cur_phase"]))
        d = Dense(4, activation="sigmoid", name="num_vec_mapping")
        dic_lane = {}
        for i, m in enumerate(self.dic_traffic_env_conf["list_lane_order"]):
            tmp_vec = d(Lambda(slice_tensor, arguments={"index": i}, name="vec_%d" % i)(dic_input_node["lane_num_vehicle"]))
            tmp_phase = Lambda(slice_tensor, arguments={"index": i}, name="phase_%d" % i)(p)
            dic_lane[m] = concatenate([tmp_vec, tmp_phase], name="lane_%d" % i)
        if self.num_actions == 8:
            list_phase_pressure = []
            lane_embedding = Dense(16, activation="relu", name="lane_embedding")
            for phase in self.dic_traffic_env_conf["PHASE"]:
                m1, m2 = phase.split("_")
                list_phase_pressure.append(add([lane_embedding(dic_lane[m1]), lane_embedding(dic_lane[m2])], name=phase))
        elif self.num_actions == 4:
            list_phase_pressure = []
            for phase in self.dic_traffic_env_conf["PHASE"]:
                m1, m2 = phase.split("_")
                list_phase_pressure.append(concatenate([dic_lane[m1], dic_lane[m2]], name=phase))

        constant = Lambda(relation, arguments={"dic_traffic_env_conf": self.dic_traffic_env_conf},
                        name="constant")(dic_input_node["lane_num_vehicle"])
        relation_embedding = Embedding(2, 4, name="relation_embedding")(constant)

        # rotate the phase pressure
        if self.dic_agent_conf["ROTATION"]:
            list_phase_pressure_recomb = []
            num_phase = self.num_phases
            for i in range(num_phase):
                for j in range(num_phase):
                    if i != j:
                        list_phase_pressure_recomb.append(
                            concatenate([list_phase_pressure[i], list_phase_pressure[j]],
                                        name="concat_compete_phase_%d_%d" % (i, j)))

            list_phase_pressure_recomb = concatenate(list_phase_pressure_recomb, name="concat_all")
            feature_map = Reshape((self.num_actions, self.num_actions-1, 32))(list_phase_pressure_recomb)
            lane_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu", name="lane_conv")(feature_map)
            if self.dic_agent_conf["MERGE"] == "multiply":
                relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                                       name="relation_conv")(relation_embedding)
                combine_feature = multiply([lane_conv, relation_conv], name="combine_feature")
            elif self.dic_agent_conf["MERGE"] == "concat":
                relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                                       name="relation_conv")(relation_embedding)
                combine_feature = concatenate([lane_conv, relation_conv], name="combine_feature")
            elif self.dic_agent_conf["MERGE"] == "weight":
                relation_conv = Conv2D(1, kernel_size=(1, 1), activation="relu", name="relation_conv")(relation_embedding)
                relation_conv = Lambda(lambda x: K.repeat_elements(x, self.dic_agent_conf["D_DENSE"], 3),
                                       name="expansion")(relation_conv)
                combine_feature = multiply([lane_conv, relation_conv], name="combine_feature")

            hidden_layer = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                                  name="combine_conv")(combine_feature)
            before_merge = Conv2D(1, kernel_size=(1, 1), activation="linear", name="befor_merge")(hidden_layer)
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(Reshape((self.num_actions, self.num_actions-1))(before_merge))

        else:
            if self.dic_agent_conf['CONFLICT_MATRIX']:
                phase_pressure = Reshape((feature_shape["lane_num_vehicle"],), name="phase_pressure")(phase_pressure)
            else:
                phase_pressure = Reshape((feature_shape["lane_num_vehicle"] * 2,), name="phase_pressure")(lane_demand)
            q_values = competing_network(phase_pressure, self.dic_agent_conf, self.num_actions)

        network = Model(inputs=[dic_input_node[feature_name]
                                for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]],
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        #network.summary()
        return network

    def convert_state_to_input(self, s):
        inputs = []
        if self.num_phases == 2:
            dic_phase_expansion = self.dic_traffic_env_conf["phase_expansion_4_lane"]
        else:
            dic_phase_expansion = self.dic_traffic_env_conf["phase_expansion"]
        for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if feature == "cur_phase":
                inputs.append(np.array([dic_phase_expansion[s[feature]+1]]))
            else:
                inputs.append(np.array([s[feature]]))
        return inputs

    def get_action(self, ob):
        if not self.if_test and np.random.rand() <= self.epsilon:
            self.action = self.action_space.sample()
            return self.action
        state={}
        state["cur_phase"] = self.world.id2intersection[self.iid].current_phase
        self.last_action = state["cur_phase"]
        state["lane_num_vehicle"] = ob
        q_values = self.model.predict(self.convert_state_to_input(state))
        self.action = np.argmax(q_values[0])
        return self.action

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        last_state = {"cur_phase": self.last_action, "lane_num_vehicle": ob}
        state = {"cur_phase": action, "lane_num_vehicle": next_ob}
        self.memory.append((self.convert_state_to_input(last_state), action, reward, self.convert_state_to_input(state)))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for input_list, action, reward, next_input in minibatch:
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_input)[0]))
            target_f = self.model.predict(input_list)
            target_f[0][action] = target
            history = self.model.fit(input_list, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/frap", e = 0):
        name = "dqn_agent_{}_{}.h5".format(self.iid, e)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/frap", e = 0):
        name = "dqn_agent_{}_{}.h5".format(self.iid, e)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)