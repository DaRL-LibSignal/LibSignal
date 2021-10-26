import gym
from environment_colight import TSCEnv
from world import World
from agent.dqn_agent import DQNAgent
from generator import LaneVehicleGenerator
from agent.presslight_agent import PressLightAgent
from metric import TravelTimeMetric
import argparse
import numpy as np
import logging
from datetime import datetime
from multiprocessing import Process, Queue
from utils import *
import pickle
from agent.colight_agent import CoLightAgent


import os

cluster_num_limit = 11
cluster_threshold = 0.2
cluster_update_rate = 10
cluster_num = 1

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str, help='path of config file')  # road net
parser.add_argument('--thread', type=int, default=8, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="0", help='gpu to be used')  # choose gpu card
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help="learning rate")
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="batch size")
parser.add_argument('-ls', '--learning_start', type=int, default=1000, help="learning start")
parser.add_argument('-rs', '--replay_buffer_size', type=int, default=5000, help="size of replay buffer")
parser.add_argument('-uf', '--update_target_model_freq', type=int, default=10,
                    help="the frequency to update target q model")
parser.add_argument('-pr', '--prefix', type=str, default="yzy1", help="the prefix of model and file")
parser.add_argument('-gc', '--grad_clip', type=float, default=5.0, help="clip gradients")
parser.add_argument('-ep', '--epsilon', type=float, default=0.8, help="exploration rate")
parser.add_argument('-ed', '--epsilon_decay', type=float, default=0.9995, help="decay rate of exploration rate")
parser.add_argument('-me', '--min_epsilon', type=float, default=0.01, help="the minimum epsilon when decaying")
parser.add_argument('--steps', type=int, default=3600, help='number of steps')  # per episodes
parser.add_argument('--test_steps', type=int, default=3600, help='number of steps for step')
parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
# parser.add_argument('--test_episodes',type=int,default=10,help='testing episodes')
parser.add_argument('--load_model_dir', type=str, default=None, help='load this model to test')
parser.add_argument('--graph_info_dir', type=str, default="syn33",
                    help='load infos about graph(i.e. mapping, adjacent)')
parser.add_argument('--train_model', action="store_false", default=True)
parser.add_argument('--test_model', action="store_true", default=False)
parser.add_argument('--save_model', action="store_false", default=True)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=10,
                    help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/colight_maml_1x5", help='directory in which model should be saved')
# parser.add_argument('--load_dir',type=str,default="model/colight",help='directory in which model should be loaded')
parser.add_argument('--log_dir', type=str, default="log/colight_maml_1x5", help='directory in which logs should be saved')
parser.add_argument('--vehicle_max', type=int, default=1, help='used to normalize node observayion')
parser.add_argument('--mask_type', type=int, default=0, help='used to specify the type of softmax')
parser.add_argument('--get_attention', action="store_true", default=False)
parser.add_argument('--test_when_train', action="store_false", default=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
file_prefix = args.prefix + "_" + "Colight_" + str(args.graph_info_dir) + "_" + str(args.learning_rate) + "_" + str(
    args.epsilon) + "_" + str(args.epsilon_decay) + "_" + str(args.batch_size) + "_" + str(
    args.learning_start) + "_" + str(args.replay_buffer_size) + "_" + datetime.now().strftime('%Y%m%d-%H%M%S')
fh = logging.FileHandler(os.path.join(args.log_dir, file_prefix + ".log"))
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

class TaskTailor:
    def __init__(self):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.layers import LSTM
        from keras.layers import Activation
        from keras.models import model_from_json
        self.samples = []
        model = Sequential()
        model.add(LSTM(units=32, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        with open('model/maml_cluster/tailor.json', 'r') as file:
            model_json = file.read()
        model = model_from_json(model_json)
        model.load_weights('model/maml_cluster/tailor.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        self.model = model

    def add_sample(self, sample):
        self.samples.append(sample)
    def divide(self):
        result = self.model.predict([[self.samples]])
        return np.argmax(result)
        # return 0
    def getlen(self):
        return len(self.samples)

world0 = World(args.config_file, thread_num=args.thread)
graph_info_file_dir = args.graph_info_dir + ".pkl"
graph_info_file = open(graph_info_file_dir, "rb")
res = pickle.load(graph_info_file)
net_node_dict_id2inter = res[0]
net_node_dict_inter2id = res[1]
net_edge_dict_id2edge = res[2]
net_edge_dict_edge2id = res[3]
node_degree_node = res[4]
node_degree_edge = res[5]
node_adjacent_node_matrix = res[6]
node_adjacent_edge_matrix = res[7]
edge_adjacent_node_matrix = res[8]
# net_node_dict_id2inter, net_node_dict_inter2id, net_edge_dict_id2edge, net_edge_dict_edge2id, \
#     node_degree_node,node_degree_edge, node_adjacent_node_matrix, node_adjacent_edge_matrix, \
#     edge_adjacent_node_matrix = pickle.load(graph_info_file)
graph_info_file.close()
# TODO:update the below dict (already done)
dic_traffic_env_conf = {
    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": len(net_node_dict_id2inter),  # used
    "NUM_ROADS": len(net_edge_dict_id2edge),  # used
    "MIN_ACTION_TIME": 10,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 8,  # used
    "NUM_LANES": 1,  # used
    "ACTION_DIM": 2,
    "MEASURE_TIME": 10,
    "IF_GUI": True,
    "DEBUG": False,
    "INTERVAL": 1,
    "THREADNUM": 8,
    "SAVEREPLAY": True,
    "RLTRAFFICLIGHT": True,
    "DIC_FEATURE_DIM": dict(  # used
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),
        D_COMING_VEHICLE=(4,),
        D_LEAVING_VEHICLE=(4,),
        D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
        D_CUR_PHASE=(1,),  # used
        D_NEXT_PHASE=(1,),
        D_TIME_THIS_PHASE=(1,),
        D_TERMINAL=(1,),
        D_LANE_SUM_WAITING_TIME=(4,),
        D_VEHICLE_POSITION_IMG=(4, 60,),
        D_VEHICLE_SPEED_IMG=(4, 60,),
        D_VEHICLE_WAITING_TIME_IMG=(4, 60,),
        D_PRESSURE=(1,),
        D_ADJACENCY_MATRIX=(2,)),
    # used
    "LIST_STATE_FEATURE": [
        "cur_phase",
        # "time_this_phase",
        # "vehicle_position_img",
        # "vehicle_speed_img",
        # "vehicle_acceleration_img",
        # "vehicle_waiting_time_img",
        "lane_num_vehicle",
        # "lane_num_vehicle_been_stopped_thres01",
        # "lane_num_vehicle_been_stopped_thres1",
        # "lane_queue_length",
        # "lane_num_vehicle_left",
        # "lane_sum_duration_vehicle_left",
        # "lane_sum_waiting_time",
        # "terminal",

        # "coming_vehicle",
        # "leaving_vehicle",
        # "pressure",

        # "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
        "sum_duration_vehicle_left": 0,
        "sum_num_vehicle_been_stopped_thres01": 0,
        "sum_num_vehicle_been_stopped_thres1": -0.25,
        "pressure": 0,
    },
    "LANE_NUM": {
        "LEFT": 1,
        "RIGHT": 1,
        "STRAIGHT": 1
    },
    "PHASE": [
        'WSES',
        'NSSS',
        'WLEL',
        'NLSL',
        'WSWL',
        'ESEL',
        'NSNL',
        'SSSL',
    ],
}

dic_graph_setting = {
    "NEIGHBOR_NUM": 4,  # standard number of adjacent nodes of each node
    "NEIGHBOR_EDGE_NUM": 4,  # # standard number of adjacent edges of each node
    "N_LAYERS": 1,  # layers of MPNN
    "INPUT_DIM": [128, 128],
    # input dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
    "OUTPUT_DIM": [128, 128],
    # output dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
    "NODE_EMB_DIM": [128, 128],  # the firsr two layer of dense to embedding the input
    "NUM_HEADS": [5, 5],
    "NODE_LAYER_DIMS_EACH_HEAD": [16, 16],  # [input_dim,output_dim]
    "OUTPUT_LAYERS": [],  #
    "NEIGHBOR_ID": node_adjacent_node_matrix,  # adjacent node id of each node
    "ID2INTER_MAPPING": net_node_dict_id2inter,  # id ---> intersection mapping
    "INTER2ID_MAPPING": net_node_dict_inter2id,  # intersection ----->id mapping
    "NODE_DEGREE_NODE": node_degree_node,  # number of adjacent nodes of node
}
class TaskTailor:
    def __init__(self):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.layers import LSTM
        from keras.layers import Activation
        from keras.models import model_from_json
        self.samples = []
        model = Sequential()
        model.add(LSTM(units=128, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        with open('model/colight_maml_cluster_new2/tailor.json', 'r') as file:
            model_json = file.read()
        model = model_from_json(model_json)
        model.load_weights('model/colight_maml_cluster_new2/tailor.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        self.model = model

    def add_sample(self, sample):
        s = []
        for k in range(len(sample)):
            s.append(np.ravel(sample[k]))
        self.samples.append(s)
    def divide(self):
        result = self.model.predict([np.reshape(self.samples, (-1, 100, 32))])
        return np.argmax(result) % 2
    def getlen(self):
        return len(self.samples)

def build(path):
    world = World(path, thread_num=args.thread)
    # create observation generator, which is used to construct sample
    observation_generators = []
    for node_dict in world.intersections:
        node_id = node_dict.id
        node_id_int = net_node_dict_inter2id[node_id]
        tmp_generator = LaneVehicleGenerator(world,
                                             node_dict, ["lane_count"],
                                             in_only=True,
                                             average='road')
        observation_generators.append((node_id_int, tmp_generator))
    sorted(observation_generators,
           key=lambda x: x[0])  # sorted the ob_generator based on its corresponding id_int, increasingly

    # create agent
    action_space = gym.spaces.Discrete(len(world.intersections[0].phases))
    colightAgent = CoLightAgent(
        action_space, observation_generators,
        LaneVehicleGenerator(world, world.intersections[0], ["lane_waiting_count"], in_only=True, average="all",
                             negative=True), world, dic_traffic_env_conf, dic_graph_setting, args)
    if args.load_model:
        colightAgent.load_model(args.load_dir)
    print(colightAgent.ob_length)
    print(colightAgent.action_space)
    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, colightAgent, metric)
    return world, colightAgent, env

def test(path, c, q):
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto(
        device_count={"CPU": 12},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    KTF.clear_session()
    world, agent, env = build(path)
    model_name = args.save_dir + "_{}".format(0)
    agent.load_model(model_name)
    attention_mat_list = []
    obs = env.reset()
    last_obs = obs
    ep_rwds = [0 for i in range(len(world.intersections))]
    eps_nums = 0
    tailor = TaskTailor()
    for i in range(args.test_steps):
        if i % args.action_interval == 0:
            last_phase = []
            for j in range(len(world.intersections)):
                node_id_str = agent.graph_setting["ID2INTER_MAPPING"][j]
                node_dict = world.id2intersection[node_id_str]
                last_phase.append(node_dict.current_phase)
            if args.get_attention:
                actions, att_step = agent.get_action(last_phase, obs, test_phase=True)
                attention_mat_list.append(att_step[0])
            else:
                actions = agent.get_action(last_phase, obs, test_phase=True)
            actions = actions[0]
            rewards_list = []
            actions[len(actions) - 1] = 0
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
                cur_phase = []
                for j in range(len(world.intersections)):
                    node_id_str = agent.graph_setting["ID2INTER_MAPPING"][j]
                    node_dict = world.id2intersection[node_id_str]
                    cur_phase.append(node_dict.current_phase)
                # if tailor.getlen() < 100:
                #     s = [list(np.mean(last_obs, axis=1))]
                #     # s.append(list(last_phase))
                #     # s.append(list(actions))
                #     s.append(list(rewards))
                #     # s.append(list(np.mean(obs, axis=1)))
                #     # s.append(list(cur_phase))
                #     tailor.add_sample(s)
                # elif tailor.getlen() == 100:
                #     print("loading")
                #     c = tailor.divide()
                #     agent.load_model(args.save_dir + "_{}".format(c + 1))
                #     s = [list(np.mean(last_obs, axis=1))]
                #     s.append(list(last_phase))
                #     s.append(list(actions))
                #     s.append(list(rewards))
                #     s.append(list(np.mean(obs, axis=1)))
                #     s.append(list(cur_phase))
                #     tailor.add_sample(s)
                last_obs = obs
            rewards = np.mean(rewards_list, axis=0)
            for j in range(len(world.intersections)):
                ep_rwds[j] += rewards[j]
            eps_nums += 1
            # ep_rwds.append(rewards)

        # print(env.eng.get_average_travel_time())
        if all(dones):
            break
    mean_rwd = np.sum(ep_rwds) / eps_nums
    trv_time = env.eng.get_average_travel_time()
    logger.info(path)
    logger.info("Final Travel Time is %.4f, and cluster %.4f" % (trv_time, c))
    q.put((path, env.eng.get_average_travel_time()))
    return trv_time


def meta_test(path, c, q):
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto(
        device_count={"CPU": 12},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    KTF.clear_session()
    world, agent, env = build(path)
    model_name = args.save_dir + "_{}".format(0)
    agent.load_model(model_name)
    attention_mat_list = []
    obs = env.reset()
    ep_rwds = [0 for i in range(len(world.intersections))]
    eps_nums = 0
    total_decision_num = 0
    episodes_rewards = [0 for i in range(len(world.intersections))]
    episodes_decision_num = 0
    episode_loss = []
    last_obs = env.reset()
    tailor = TaskTailor()
    for i in range(args.test_steps):
        if i % args.action_interval == 0:
            actions = []
            last_phase = []  # ordered by the int id of intersections
            for j in range(len(world.intersections)):
                node_id_str = agent.graph_setting["ID2INTER_MAPPING"][j]
                node_dict = world.id2intersection[node_id_str]
                last_phase.append(node_dict.current_phase)
                # last_phase.append([self.world.intersections[j].current_phase])
            actions = agent.get_action(last_phase, last_obs, test_phase=True)
            actions = actions[0]
            reward_list = []  # [intervals,agents,reward]
            actions[len(actions) - 1] = 0
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
                reward_list.append(rewards)
                # rewards = np.mean(reward_list, axis=0)
                for j in range(len(world.intersections)):
                    episodes_rewards[j] += rewards[j]
                cur_phase = []
                for j in range(len(world.intersections)):
                    node_id_str = agent.graph_setting["ID2INTER_MAPPING"][j]
                    node_dict = world.id2intersection[node_id_str]
                    cur_phase.append(node_dict.current_phase)
                last_phase = []  # ordered by the int id of intersections
                for j in range(len(world.intersections)):
                    node_id_str = agent.graph_setting["ID2INTER_MAPPING"][j]
                    node_dict = world.id2intersection[node_id_str]
                    last_phase.append(node_dict.current_phase)
                    # last_phase.append([self.world.intersections[j].current_phase])
                agent.remember(last_obs, last_phase, actions, rewards, obs, cur_phase)
                # if tailor.getlen() < 100:
                #     s = [list(np.mean(last_obs, axis=1))]
                #     # s.append(list(last_phase))
                #     # s.append(list(actions))
                #     s.append(list(rewards))
                #     # s.append(list(np.mean(obs, axis=1)))
                #     # s.append(list(cur_phase))
                #     tailor.add_sample(s)
                # elif tailor.getlen() == 100:
                #     print("loading")
                #     c = tailor.divide()
                #     agent.load_model(args.save_dir + "_{}".format(c + 1))
                #     s = [list(np.mean(last_obs, axis=1))]
                #     s.append(list(last_phase))
                #     s.append(list(actions))
                #     s.append(list(rewards))
                #     s.append(list(np.mean(obs, axis=1)))
                #     s.append(list(cur_phase))
                #     tailor.add_sample(s)
                episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs

                if total_decision_num > 500 and total_decision_num % 20 == 20 - 1:
                    cur_loss_q = agent.replay()
                    episode_loss.append(cur_loss_q)
                if total_decision_num > 500 and total_decision_num % 400 == 400 - 1:
                    agent.update_target_network()
        if all(dones):
            break
    mean_rwd = np.sum(ep_rwds) / eps_nums
    trv_time = env.eng.get_average_travel_time()
    t1 = trv_time
    # logger.info("Final Travel Time is %.4f, and mean rewards %.4f" % (trv_time, mean_rwd))
    attention_mat_list = []
    # print(total_decision_num)
    env.reset()
    for i in range(args.test_steps):
        if i % args.action_interval == 0:
            last_phase = []
            for j in range(len(world.intersections)):
                node_id_str = agent.graph_setting["ID2INTER_MAPPING"][j]
                node_dict = world.id2intersection[node_id_str]
                last_phase.append(node_dict.current_phase)
            if args.get_attention:
                actions, att_step = agent.get_action(last_phase, obs, test_phase=True)
                attention_mat_list.append(att_step[0])
            else:
                actions = agent.get_action(last_phase, obs, test_phase=True)
            actions = actions[0]
            rewards_list = []
            actions[len(actions) - 1] = 0
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
            rewards = np.mean(rewards_list, axis=0)
            # ep_rwds.append(rewards)

        # print(env.eng.get_average_travel_time())
        if all(dones):
            break
    logger.info(path)
    logger.info("Final Travel Time is %.4f, and cluster %.4f" % (t1, c))
    logger.info("Meta Test Result: Final Travel Time is %.4f" % env.eng.get_average_travel_time())
    t2 = env.eng.get_average_travel_time()
    q.put((path, np.minimum(t1, t2)))
    return np.minimum(t1, t2)


if __name__ == '__main__':
    # meta_train(args)
    # train(args)
    real_flow_path = []
    p_path = ['/mnt/c/users/onlyc/desktop/work/RRL_TLC/real_flow/']
    real_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/flow_config_1x5/0/'
    for root, dirs, files in os.walk(real_flow_floder):
        for file in files:
            real_flow_path.append(real_flow_floder + file)
    logger.info("Meta Test Real")
    result = []
    q = Queue()
    pool = []
    # meta_test(real_flow_path[0], 0, q)
    for n in range(len(real_flow_path)):
        for c in range(cluster_num):
            p = Process(target=test, args=(real_flow_path[n], c, q))
            pool.append(p)
            p.start()
            p = Process(target=meta_test, args=(real_flow_path[n], c, q))
            pool.append(p)
            p.start()
    for p in pool:
        p.join()
    result_dict = {}
    for i in range(len(real_flow_path) * cluster_num * 2):
        item = q.get()
        if item[0] not in result_dict:
            result_dict[item[0]] = item[1]
        else:
            result_dict[item[0]] = np.minimum(item[1], result_dict[item[0]])
    for i in range(len(real_flow_path)):
        result.append(result_dict[real_flow_path[i]])
    logger.info(
        "Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
    fake_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/flow_config_1x5/'
    w_dis = [0.005, 0.01, 0.05, 0.1]
    # w_dis = [0.05]
    for w in w_dis:
        pool = []
        q = Queue()
        logger.info("Meta Test Fake with W Distance: %.4f" % w)
        fake_flow_path = []
        result = []
        for root, dirs, files in os.walk(fake_flow_floder + str(w) + '/'):
            for file in files:
                fake_flow_path.append(fake_flow_floder + str(w) + '/' + file)
        for n in range(len(fake_flow_path)):
            logger.info("Meta Test Env: %d" % n)
            for c in range(cluster_num):
                p = Process(target=test, args=(fake_flow_path[n], c, q))
                pool.append(p)
                p.start()
                p = Process(target=meta_test, args=(fake_flow_path[n], c, q))
                pool.append(p)
                p.start()
        for p in pool:
            p.join()
        result_dict = {}
        for i in range(len(fake_flow_path) * cluster_num * 2):
            item = q.get()
            if item[0] not in result_dict:
                result_dict[item[0]] = item[1]
            else:
                result_dict[item[0]] = np.minimum(item[1], result_dict[item[0]])
        for i in range(len(fake_flow_path)):
            result.append(result_dict[fake_flow_path[i]])
        logger.info(
            "Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
