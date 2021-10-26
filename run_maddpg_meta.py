import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.maddpg_agent import MADDPGAgent
from metric import TravelTimeMetric
import argparse
import tensorflow as tf
import os
import logging
from datetime import datetime
import numpy as np


# parse args
def parse_args():
    parser = argparse.ArgumentParser(description='Run Example')
    # Environment
    parser.add_argument('config_file', type=str, help='path of config file')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
    parser.add_argument('--episodes', type=int, default=500, help='training episodes')
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="model/maddpg", help="directory in which model should be saved")
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument('--log_dir', type=str, default="log/maddpg", help='directory in which logs should be saved')
    return parser.parse_args()


args = parse_args()
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


def build(path):
    # create world
    world = World(path, thread_num=args.thread)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(MADDPGAgent(
            action_space,
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
            args,
            i.id
        ))
    ob_space_n = []
    action_space_n = []
    for agent in agents:
        ob_space_n.append(agent.ob_shape)
        action_space_n.append(agent.action_space)
    print(ob_space_n)
    print(action_space_n)
    for i, agent in enumerate(agents):
        agent.build_model(ob_space_n, action_space_n, i)

    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, agents, metric)
    return world, agents, env


# train maddpg_agent
def meta_train(path):
    world, agents, env = build(args.config_file)
    # meta_world, meta_agents, meta_env = [], [], []
    # for n in range(len(path)):
    #     w, a, e = build(path[n])
    #     meta_world.append(w)
    #     meta_agents.append(a)
    #     meta_env.append(e)
    config = tf.ConfigProto(
        intra_op_parallelism_threads=4,
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    with sess:
        # Initialize
        sess.run(tf.variables_initializer(tf.global_variables()))
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        train_step = 0

        print('Starting iterations...')
        for e in range(args.episodes):
            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
            obs_n = env.reset()
            episode_step = 0
            step = 0
            while step < args.steps:
                if step % args.action_interval == 0:
                    # get action
                    action_n = [agent.get_action(obs) for agent, obs in zip(agents, obs_n)]
                    action_prob_n = [agent.get_action_prob(obs) for agent, obs in zip(agents, obs_n)]
                    # environment step
                    for _ in range(args.action_interval):
                        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                        step += 1

                    episode_step += 1
                    # collect experience
                    for i, agent in enumerate(agents):
                        agent.experience(obs_n[i], action_prob_n[i], rew_n[i], new_obs_n[i], done_n[i])
                    obs_n = new_obs_n

                    for i, rew in enumerate(rew_n):
                        episode_rewards[-1] += rew
                        agent_rewards[i][-1] += rew

                    # increment global step counter
                    train_step += 1

                    # update all trainers, if not in display or benchmark mode
                    loss = None
                    for agent in agents:
                        agent.update(agents, train_step)
                        # print(loss)
                        # if loss is not None:
                        #     print(loss[0], loss[1])

            # logger.info("episode:{}/{}, total agent episode mean reward:{}".format(e, args.episodes,
            #                                                                  episode_rewards[0] / episode_step))
            logger.info(
                "episode:{}/{}, average travel time:{}".format(e, args.episodes, env.eng.get_average_travel_time()))
            for i in range(len(agents)):
                logger.info("agent:{}, episode mean reward:{}".format(i, agent_rewards[i][-1] / episode_step))
            if e % args.save_rate == 0:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                saver.save(sess, os.path.join(args.save_dir, "maddpg_{}.ckpt".format(e)))


def test(path, model_id=None):
    tf.reset_default_graph()
    world, agents, env = build(path)
    sess = tf.Session()
    with sess:
        # Initialize
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(args.save_dir)
        if model_id is not None:
            saver.restore(sess, "model/maddpg/maddpg_{}.ckpt".format(model_id))
        else:
            saver.restore(sess, model_file)
        obs_n = env.reset()
        for i in range(args.steps):
            if i % args.action_interval == 0:
                # get action
                action_n = [agent.get_action(obs) for agent, obs in zip(agents, obs_n)]
                # environment step
                for _ in range(args.action_interval):
                    obs_n, rew_n, done_n, info_n = env.step(action_n)
                done = all(done_n)
                if done:
                    break
        logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())
        return env.eng.get_average_travel_time()


def meta_test(path, model_id=None):
    tf.reset_default_graph()
    world, agents, env = build(path)
    sess = tf.Session()
    with sess:
        # Initialize
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(args.save_dir)
        if model_id is not None:
            saver.restore(sess, "model/maddpg/maddpg_{}.ckpt".format(model_id))
        else:
            saver.restore(sess, model_file)
        obs_n = env.reset()
        for i in range(args.steps):
            if i % args.action_interval == 0:
                # get action
                action_n = [agent.get_action(obs) for agent, obs in zip(agents, obs_n)]
                # environment step
                for _ in range(args.action_interval):
                    obs_n, rew_n, done_n, info_n = env.step(action_n)
                done = all(done_n)
                if done:
                    break
        logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())
    return env.eng.get_average_travel_time()


if __name__ == '__main__':
    meta_train(args)
    # train(args)
    real_flow_path = []
    real_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/real_flow_config/'
    for root, dirs, files in os.walk(real_flow_floder):
        for file in files:
            real_flow_path.append(real_flow_floder + file)
    # meta_train(real_flow_path)
    logger.info("Meta Test Real")
    result = []
    for n in range(len(real_flow_path)):
        logger.info("Meta Test Env: %d" % n)
        t1 = test(real_flow_path[n])
        t2 = meta_test(real_flow_path[n])
        result.append(np.minimum(t1, t2))
    logger.info(
        "Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
    fake_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/fake_flow_config/'
    w_dis = [0.005, 0.01, 0.05, 0.1]
    for w in w_dis:
        logger.info("Meta Test Fake with W Distance: %.4f" % w)
        fake_flow_path = []
        result = []
        for root, dirs, files in os.walk(fake_flow_floder + str(w) + '/'):
            for file in files:
                fake_flow_path.append(fake_flow_floder + str(w) + '/' + file)
        for n in range(len(fake_flow_path)):
            logger.info("Meta Test Env: %d" % n)
            t1 = test(fake_flow_path[n])
            t2 = meta_test(fake_flow_path[n])
            result.append(np.minimum(t1, t2))
        logger.info(
            "Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
