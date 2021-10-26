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

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=4, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=True)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=20,
                    help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/meta_presslight_4x4",
                    help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/meta_presslight_4x4",
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


def build(path):
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
    return world, agents, env


# train presslight_agent
def train(args):
    world, agents, env = build(args.config_file)
    total_decision_num = 0
    for e in range(args.episodes):
        last_obs = env.reset()
        # if e % args.save_rate == args.save_rate - 1:
        #     env.eng.set_save_replay(True)
        #     env.eng.set_replay_file("replay_%s.txt" % e)
        # else:
        #     env.eng.set_save_replay(False)
        episodes_rewards = [0 for i in agents]
        episodes_decision_num = 0
        i = 0
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
        # if e % args.save_rate == args.save_rate - 1:
        #     if not os.path.exists(args.save_dir):
        #         os.makedirs(args.save_dir)
        #     for agent in agents:
        #         agent.save_model(args.save_dir)
        logger.info("episode:{}/{}, average travel time:{}".format(e, args.episodes, env.eng.get_average_travel_time()))
        for agent_id, agent in enumerate(agents):
            logger.info(
                "agent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))


def meta_train(path):
    meta_world, meta_agents, meta_env = [], [], []
    for n in range(len(path)):
        w, a, e = build(path[n])
        meta_world.append(w)
        meta_agents.append(a)
        meta_env.append(e)
    total_decision_num = 0
    for e in range(args.episodes):
        meta_last_obs = []
        meta_episodes_rewards = []
        meta_episodes_decision_num = []
        for n in range(len(path)):
            meta_last_obs.append(meta_env[n].reset())
            meta_episodes_rewards.append([0 for i in meta_agents[n]])
            meta_episodes_decision_num.append(0)
        i = 0
        while i < args.steps:
            # print(i)
            j = i
            for n in range(len(path)):
                if i % args.action_interval == 0:
                    actions = []
                    last_phase = []
                    for agent_id, agent in enumerate(meta_agents[0]):
                        last_phase.append([meta_env[n].world.intersections[agent_id].current_phase])
                        if total_decision_num > agent.learning_start:
                            # pif True:
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

                    for agent_id, agent in enumerate(meta_agents[0]):
                        agent.remember(meta_last_obs[n][agent_id], last_phase[agent_id], actions[agent_id],
                                       rewards[agent_id],
                                       obs[agent_id],
                                       [meta_env[n].world.intersections[agent_id].current_phase])
                        meta_episodes_rewards[n][agent_id] += rewards[agent_id]
                        meta_episodes_decision_num[n] += 1
                    total_decision_num += 1

                    meta_last_obs[n] = obs
                if n < len(path) - 1:
                    i = j
            for agent_id, agent in enumerate(meta_agents[0]):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
            if all(dones):
                break
        if e % args.save_rate == args.save_rate - 1:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            for agent in meta_agents[0]:
                agent.save_model(args.save_dir, e)
        logger.info("episode:{}/{}".format(e, args.episodes))
        for n in range(len(path)):
            logger.info("env:{}, average travel time:{}".format(n, meta_env[
                n].eng.get_average_travel_time()))
        # for agent_id, agent in enumerate(meta_agents[0]):
        #     for n in range(len(path)):
        #         logger.info(
        #             "env:{}, agent:{}, mean_episode_reward:{}".format(n, agent_id,
        #                                                               meta_episodes_rewards[n][agent_id] /
        #                                                               meta_episodes_decision_num[n]))


def test(path):
    world, agents, env = build(path)
    obs = env.reset()
    for agent in agents:
        agent.load_model(args.save_dir)
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
    return env.eng.get_average_travel_time()

def meta_test(path):
    world, agents, env = build(path)
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
            for _ in range(args.action_interval):
                rewards_list = []
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
    print(total_decision_num)
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
    return np.minimum(t1, t2)


if __name__ == '__main__':
    # meta_train(args)
    # train(args)
    real_flow_path = []
    real_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/flow_config_4x4/0/'
    for root, dirs, files in os.walk(real_flow_floder):
        for file in files:
            real_flow_path.append(real_flow_floder + file)
    meta_train(real_flow_path)
    logger.info("Meta Test Real")
    result = []
    for n in range(len(real_flow_path)):
        logger.info("Meta Test Env: %d" % n)
        t1 = test(real_flow_path[n])
        t2 = meta_test(real_flow_path[n])
        result.append(np.minimum(t1, t2))
    logger.info(
        "Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
    fake_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/flow_config_4x4/'
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
        logger.info("Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
