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
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

cluster_num_limit = 11
cluster_threshold = 0.2
cluster_update_rate = 10
cluster_update_start = 50 # at least 10
# cluster_num = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
parser.add_argument('--save_dir', type=str, default="model/maml_cluster_case",
                    help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/maml_cluster_case",
                    help='directory in which logs should be saved')
parser.add_argument('--cluster_num', type=int, default=1,
                    help='number of clusters')
args = parser.parse_args()

cluster_num = args.cluster_num
args.log_dir += '_{}'.format(cluster_num)
args.save_dir += '_{}'.format(cluster_num)
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


def build(path, is_virtual=False):
    # create world
    world = World(path, thread_num=args.thread)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(PressLightAgent(
            action_space,
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
            i.id,
            world,
            is_virtual
        ))
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


def meta_train(path, p_path):
    embeddings = []
    # for p in range(11):
    #     embeddings.append(np.load(p_path[0] + 'real_flow_' + str(p) + '.npy'))
    meta_world, meta_agents, meta_env = [], [], []
    accumulate_reward = []
    total_decision_num = []
    is_virtual = False
    samples = []
    for n in range(len(path) * 2):
        if n == len(path):
            is_virtual = True
        w, a, e = build(path[n % len(path)], is_virtual)
        meta_world.append(w)
        meta_agents.append(a)
        meta_env.append(e)
        total_decision_num.append(0)
        accumulate_reward.append(0)
    key_worlds, key_agents, key_envs = [], [], []
    for i in range(cluster_num):
        key_world, key_agent, key_env = build(path[0])
        key_worlds.append(key_world)
        key_agents.append(key_agent)
        key_envs.append(key_env)
    key_num = np.zeros((cluster_num, 1))
    env2cluster = []
    for i in range(len(path)):
        env2cluster.append(i % cluster_num)
    for e in range(args.episodes):
        sample = []
        for n in range(len(path)):
            sample.append([])
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        for c in range(cluster_num):
            for agent in key_agents[c]:
                agent.save_model(args.save_dir, c, e)
        meta_last_obs = []
        meta_episodes_rewards = []
        meta_episodes_decision_num = []
        for n in range(len(path) * 2):
            meta_last_obs.append(meta_env[n].reset())
            meta_episodes_rewards.append([0 for i in meta_agents[n]])
            meta_episodes_decision_num.append(0)

        # step 1

        for n in range(len(path)):
            for agent in meta_agents[n]:
                agent.load_model(args.save_dir, env2cluster[n], e)
            i = 0
            while i < args.steps:
                if i % args.action_interval == 0:

                    actions = []
                    last_phase = []
                    for agent_id, agent in enumerate(meta_agents[n]):
                        last_phase.append([meta_env[n].world.intersections[agent_id].current_phase])
                        if total_decision_num[n] > agent.learning_start:
                            # if True:
                            actions.append(
                                agent.get_action([meta_env[n].world.intersections[agent_id].current_phase],
                                                 meta_last_obs[n][agent_id]))
                        else:
                            actions.append(agent.sample())

                    rewards_list = []
                    for _ in range(args.action_interval):
                        obs, rewards, dones, _ = meta_env[n].step(actions)
                        i += 1
                        rewards_list.append(rewards)
                    rewards = np.mean(rewards_list, axis=0)

                    for agent_id, agent in enumerate(meta_agents[n]):
                        agent.remember(meta_last_obs[n][agent_id], last_phase[agent_id], actions[agent_id],
                                       rewards[agent_id],
                                       obs[agent_id],
                                       [meta_env[n].world.intersections[agent_id].current_phase])
                        if len(sample[n]) < agent.meta_test_start:
                            s = list(meta_last_obs[n][agent_id])
                            s.append(last_phase[agent_id][0])
                            s.append(actions[agent_id])
                            s.append(rewards[agent_id])
                            for ob in obs[agent_id]:
                                s.append(ob)
                            s.append(meta_env[n].world.intersections[agent_id].current_phase)
                            sample[n].append(s)
                        meta_episodes_rewards[n][agent_id] += rewards[agent_id]
                        meta_episodes_decision_num[n] += 1
                        total_decision_num[n] += 1

                    meta_last_obs[n] = obs
                    for agent_id, agent in enumerate(meta_agents[n]):
                        if total_decision_num[n] > agent.learning_start and total_decision_num[
                            n] % agent.update_model_freq == agent.update_model_freq - 1:
                            agent.replay()
                        if total_decision_num[n] > agent.learning_start and total_decision_num[
                            n] % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                            agent.update_target_network()
        # step 2
        i = 0
        while i < args.steps:
            j = i
            if i % args.action_interval == 0:
                for n in range(len(path)):
                    actions = []
                    last_phase = []
                    for agent_id, agent in enumerate(meta_agents[n]):
                        last_phase.append([meta_env[n + len(path)].world.intersections[agent_id].current_phase])
                        if total_decision_num[n] > agent.learning_start:
                            # if True:
                            actions.append(
                                agent.get_action([meta_env[n + len(path)].world.intersections[agent_id].current_phase],
                                                 meta_last_obs[n + len(path)][agent_id]))
                        else:
                            actions.append(agent.sample())

                    rewards_list = []
                    for _ in range(args.action_interval):
                        obs, rewards, dones, _ = meta_env[n + len(path)].step(actions)
                        i += 1
                        rewards_list.append(rewards)
                    rewards = np.mean(rewards_list, axis=0)

                    for agent_id, agent in enumerate(key_agents[env2cluster[n]]):
                        # if total_decision_num[n] < agent.learning_start:
                        #     continue
                        agent.remember(meta_last_obs[n + len(path)][agent_id], last_phase[agent_id], actions[agent_id],
                                       rewards[agent_id],
                                       obs[agent_id],
                                       [meta_env[n + len(path)].world.intersections[agent_id].current_phase])
                        meta_episodes_rewards[n + len(path)][agent_id] += rewards[agent_id]
                        meta_episodes_decision_num[n + len(path)] += 1
                        total_decision_num[n + len(path)] += 1
                        key_num[env2cluster[n]]+= 1

                    meta_last_obs[n + len(path)] = obs
                    if n < len(path) - 1:
                        i = j
                for c in range(cluster_num):
                    for agent_id, agent in enumerate(key_agents[c]):
                        if key_num[c] / cluster_num > agent.learning_start and key_num[c] % agent.update_model_freq == agent.update_model_freq - 1:
                            agent.replay()
                        if key_num[c] / cluster_num> agent.learning_start and key_num[c] % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                            agent.update_target_network()
            if all(dones):
                break
        # logger.info("Step 1:")
        logger.info("episode:{}/{}".format(e, args.episodes))
        for n in range(len(path)):
            logger.info("env:{},  cluster:{}, average travel time:{}".format(n, env2cluster[n],
                                                                             meta_env[n].eng.get_average_travel_time()))
            if e > cluster_update_start - cluster_update_rate:
                accumulate_reward[n] += meta_env[n].eng.get_average_travel_time()
                # accumulate_reward[n] += np.mean(samples[n])
        if e > cluster_update_start and e % cluster_update_rate == cluster_update_rate - 1:
            cluster_center = []
            for c in range(cluster_num):
                result = []
                for n in range(len(path)):
                    if env2cluster[n] == c:
                        result.append(accumulate_reward[n] / cluster_update_rate)
                cluster_center.append(np.median(result))
            new_cluster = []
            for n in range(len(path)):
                min_dis = 10000
                min_cluster = -1
                for c in range(cluster_num):
                    if min_dis > np.abs(accumulate_reward[n] / cluster_update_rate - cluster_center[c]):
                        min_dis = np.abs(accumulate_reward[n] / cluster_update_rate - cluster_center[c])
                        min_cluster = c
                env2cluster[n] = min_cluster
                accumulate_reward[n] = 0
            logger.info("===================================")
            logger.info("Update Cluster")
            logger.info(env2cluster)
            logger.info("===================================")
        # samples.append(sample)
        np.save(args.save_dir + '/samples_{}.npy'.format(e), sample)
        np.save(args.save_dir + '/env2cluster.npy', env2cluster)
        # logger.info("Step 2:")
        # for n in range(len(path)):
        #     logger.info("env:{}, episode:{}/{}, average travel time:{}".format(n, e, args.episodes, meta_env[
        #         n + len(path)].eng.get_average_travel_time()))
        # for agent_id, agent in enumerate(key_agent):
        #     for n in range(len(path)):
        #         logger.info(
        #             "env:{}, agent:{}, mean_episode_reward:{}".format(n, agent_id,
        #                                                               meta_episodes_rewards[n + len(path)][agent_id] /
        #                                                               meta_episodes_decision_num[n + len(path)]))


def test(path, c):
    world, agents, env = build(path)
    obs = env.reset()
    for agent in agents:
        agent.load_model(args.save_dir, c)
    for i in range(args.steps):
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action([env.world.intersections[agent_id].current_phase], obs[agent_id]))
            rewards_list = []
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
            rewards = np.mean(rewards_list, axis=0)
        # print(env.eng.get_average_travel_time())
        if all(dones):
            break
    logger.info("Test Result: Final Travel Time is %.4f" % env.eng.get_average_travel_time())
    # q.put((path, env.eng.get_average_travel_time()))
    return env.eng.get_average_travel_time()


def meta_test(path, c):
    world, agents, env = build(path)
    obs = env.reset()
    last_obs = obs
    # env.change_world(World(config, thread_num=args.thread))
    for agent in agents:
        agent.load_model(args.save_dir, c)
    total_decision_num = 0
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
    # q.put((path, np.minimum(t1, t2)))
    return np.minimum(t1, t2)


if __name__ == '__main__':
    # meta_train(args)
    # train(args)
    real_flow_path = []
    p_path = ['/mnt/c/users/onlyc/desktop/work/RRL_TLC/real_flow/']
    real_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/real_flow_config/'
    for root, dirs, files in os.walk(real_flow_floder):
        for file in files:
            real_flow_path.append(real_flow_floder + file)
    meta_train(real_flow_path, p_path)
    logger.info("Meta Test Real")
    result = []
    for n in range(len(real_flow_path)):
        r = 10000
        for c in range(cluster_num):
            t1 = test(real_flow_path[n], c)
            t2 = meta_test(real_flow_path[n], c)
            r = np.minimum(np.minimum(t1, t2), r)
        result.append(r)
    logger.info(
        "Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
    # fake_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/flow_config_1x5/'
    # w_dis = [0.005, 0.01, 0.05, 0.1]
    # for w in w_dis:
    #     logger.info("Meta Test Fake with W Distance: %.4f" % w)
    #     fake_flow_path = []
    #     result = []
    #     for root, dirs, files in os.walk(fake_flow_floder + str(w) + '/'):
    #         for file in files:
    #             fake_flow_path.append(fake_flow_floder + str(w) + '/' + file)
    #     for n in range(len(fake_flow_path)):
    #         logger.info("Meta Test Env: %d" % n)
    #         r = 10000
    #         for c in range(cluster_num):
    #             t1 = test(fake_flow_path[n], c)
    #             t2 = meta_test(fake_flow_path[n], c)
    #             r = np.minimum(np.minimum(t1, t2), r)
    #         result.append(r)
    #     logger.info(
    #         "Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
