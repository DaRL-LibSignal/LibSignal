import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from generator import IntersectionVehicleGenerator
from agent.presslight_agent import PressLightAgent
from metric import TravelTimeMetric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
config = tf.ConfigProto(
        device_count={"CPU": 12},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        allow_soft_placement=True,
        log_device_placement=False
    )
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=20,
                    help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/presslight_4x4", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/presslight_4x4", help='directory in which logs should be saved')
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

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(PressLightAgent(
        action_space,
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
        i.id,
        world
    ))
    if args.load_model:
        agents[-1].load_model(args.save_dir)
    # if len(agents) == 5:
    #     break
print(agents[0].ob_length)
print(agents[0].action_space)

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)


# train presslight_agent
def train(args, env):
    total_decision_num = 0
    for e in range(args.episodes):
        last_obs = env.reset()
        if e % args.save_rate == args.save_rate - 1:
            env.eng.set_save_replay(True)
            env.eng.set_replay_file("replay_%s.txt" % e)
        else:
            env.eng.set_save_replay(False)
        episodes_rewards = [0 for i in agents]
        episodes_decision_num = 0
        i = 0
        # key_agent = agents[0]
        while i < args.steps:
            if i % args.action_interval == 0:
                actions = []
                last_phase = []
                for agent_id, agent in enumerate(agents):
                    last_phase.append([env.world.intersections[agent_id].current_phase])
                    if total_decision_num > agent.learning_start:
                        # if True:
                        actions.append(
                            agent.get_action([env.world.intersections[agent_id].current_phase], last_obs[agent_id]))
                    else:
                        actions.append(agent.sample())

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
                    episodes_rewards[agent_id] += rewards[agent_id]
                    episodes_decision_num += 1
                total_decision_num += 1

                last_obs = obs
            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
            if all(dones):
                break
        if e % args.save_rate == args.save_rate - 1:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            for agent in agents:
                agent.save_model(args.save_dir)
        logger.info("episode:{}/{}, average travel time:{}".format(e, args.episodes, env.eng.get_average_travel_time()))
        for agent_id, agent in enumerate(agents):
            logger.info(
                "agent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))


def test():
    obs = env.reset()
    for agent in agents:
        agent.load_model(args.save_dir, 0)
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
    logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())


def meta_test(config):
    obs = env.reset()
    last_obs = obs
    # env.change_world(World(config, thread_num=args.thread))
    for agent in agents:
        agent.load_model(args.save_dir)
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
        for agent_id, agent in enumerate(agents):
            if total_decision_num > 10 and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                agent.replay()
            if total_decision_num > 10 and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                agent.update_target_network()
        # print(env.eng.get_average_travel_time())
        if all(dones):
            break
    print(total_decision_num)
    logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())
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
    logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())


if __name__ == '__main__':
    # simulate
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    train(args, env)
    test()
    # meta_test('/mnt/d/Cityflow/examples/config.json')