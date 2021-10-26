import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.presslight_agent import PressLightAgent
from metric import TravelTimeMetric
import argparse
import numpy as np
import logging
from datetime import datetime
from multiprocessing import Process, Queue


import os

cluster_num_limit = 11
cluster_threshold = 0.2
cluster_update_rate = 10
cluster_num = 1

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=True)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=1,
                    help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/meta_presslight_4x4/",
                    help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/meta_presslight_4x4/",
                    help='directory in which logs should be saved')
args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S') + ".log"))
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
    def getlen(self):
        return len(self.samples)

def build(path, is_virtual=False):
    # create world
    world = World(path, thread_num=args.thread)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        agent = PressLightAgent(
            action_space,
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
            i.id,
            world,
            is_virtual
        )
        agent.epsilon = agent.epsilon_min
        agents.append(agent)
        if args.load_model:
            agents[-1].load_model(args.save_dir)
        # if len(agents) == 5:
        #     break
    # print(agents[0].ob_length)
    # print(agents[0].action_space)

    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, agents, metric)
    return world, agents, env

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
    world, agents, env = build(path)
    obs = env.reset()
    last_obs = obs
    tailor = TaskTailor()
    total_decision_num = 0
    for agent in agents:
        agent.load_model(args.save_dir, 119, 0)
    for i in range(args.steps):
        if i % args.action_interval == 0:
            last_phase = []
            actions = []
            for agent_id, agent in enumerate(agents):
                last_phase.append([env.world.intersections[agent_id].current_phase])
                actions.append(agent.get_action([env.world.intersections[agent_id].current_phase], obs[agent_id]))
            rewards_list = []
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
                for agent_id, agent in enumerate(agents):
                    agent.remember(last_obs[agent_id], last_phase[agent_id], actions[agent_id], rewards[agent_id],
                                   obs[agent_id],
                                   [env.world.intersections[agent_id].current_phase])
                    total_decision_num += 1
            rewards = np.mean(rewards_list, axis=0)
        # print(env.eng.get_average_travel_time())
        if all(dones):
            break
    logger.info("Test Result: Final Travel Time is %.4f" % env.eng.get_average_travel_time())
    q.put((path, env.eng.get_average_travel_time()))
    return env.eng.get_average_travel_time()


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
    world, agents, env = build(path)
    obs = env.reset()
    last_obs = obs
    # env.change_world(World(config, thread_num=args.thread))
    for agent in agents:
        agent.load_model(args.save_dir, 119, 0)
    total_decision_num = 0
    tailor = TaskTailor()
    for i in range(args.steps):
        if i % args.action_interval == 0:
            actions = []
            last_phase = []
            for agent_id, agent in enumerate(agents):
                last_phase.append([env.world.intersections[agent_id].current_phase])
                actions.append(agent.get_action([env.world.intersections[agent_id].current_phase], obs[agent_id]))
            rewards_list = []
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                for agent_id, agent in enumerate(agents):
                    agent.remember(last_obs[agent_id], last_phase[agent_id], actions[agent_id], rewards[agent_id],
                                   obs[agent_id],
                                   [env.world.intersections[agent_id].current_phase])
                    total_decision_num += 1
                last_obs = obs
                last_phase = []
                for agent_id, agent in enumerate(agents):
                    last_phase.append([env.world.intersections[agent_id].current_phase])
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.meta_test_start and total_decision_num % agent.meta_test_update_model_freq == agent.meta_test_update_model_freq - 1:
                        agent.replay()
                    if total_decision_num > agent.meta_test_start and total_decision_num % agent.meta_test_update_target_model_freq == agent.meta_test_update_target_model_freq - 1:
                        agent.update_target_network()
        # print(env.eng.get_average_travel_time())
        if all(dones):
            break
    # print(total_decision_num)
    logger.info("Meta Test: Final Travel Time is %.4f" % env.eng.get_average_travel_time())
    t1 = env.eng.get_average_travel_time()
    obs = env.reset()
    # env.eng.set_save_replay(True)
    # env.eng.set_replay_file("replay_test_0.05.txt")
    for i in range(args.steps):
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action([env.world.intersections[agent_id].current_phase], obs[agent_id]))
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
        # print(env.eng.get_average_travel_time(), env.eng.get_vehicle_count())
        if all(dones):
            break
    logger.info("Meta Test Result: Final Travel Time is %.4f" % env.eng.get_average_travel_time())
    t2 = env.eng.get_average_travel_time()
    q.put((path, np.minimum(t1, t2)))
    return np.minimum(t1, t2)


if __name__ == '__main__':
    # meta_train(args)
    # train(args)
    real_flow_path = []
    p_path = ['/mnt/c/users/onlyc/desktop/work/RRL_TLC/real_flow/']
    real_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/flow_config_4x4/0/'
    for root, dirs, files in os.walk(real_flow_floder):
        for file in files:
            real_flow_path.append(real_flow_floder + file)
    logger.info("Meta Test Real")
    result = []
    q = Queue()
    pool = []
    # re = test(real_flow_path[0], 0, q)
    # meta_test(real_flow_path[0], 0, q)
    # print(re)
    for n in range(len(real_flow_path)):
        for c in range(cluster_num):
            p = Process(target=test, args=(real_flow_path[n], c, q))
            pool.append(p)
            p.start()
            p = Process(target=meta_test, args=(real_flow_path[n], c, q))
            pool.append(p)
            p.start()
    for p in pool:
        p.join(100)
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
    fake_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/flow_config_4x4/'
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
            p.join(100)
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
