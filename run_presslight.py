import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from agent.presslight_agent import PressLightAgent
from metric import TravelTimeMetric
import argparse
import os
import numpy as np
import logging
from datetime import datetime

import time

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=20,
                    help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/presslight_1X6/torch",
                    help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/presslight_1X6/torch",
                    help='directory in which logs should be saved')
# k segmentations of input roads, default is 1                          add this feature later
parser.add_argument('--k', type=int, default=1, help='k segmentations of input roads')
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
        [
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=False, average=None),
            IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),

        ],
        # sum(w(l,m))
        LaneVehicleGenerator(world, i, ["pressure"], average="all", negative=True),
        i.id,
    ))
    if args.load_model:
        agents[-1].load_model(args.save_dir)
    # if len(agents) == 5:
    #     break

for agent in agents:
    print(agent.action_space)

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)


# train presslight_agent
def train(args, env):
    total_decision_num = 0
    timedic = {}
    startepoch = time.time() - starttime
    timedic.update({-1: startepoch})
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
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        # if True:
                        actions.append(agent.choose(last_obs[agent_id]))
                    else:
                        actions.append(agent.sample())

                rewards_list = []
                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)

                for agent_id, agent in enumerate(agents):
                    agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
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

        endepoch = time.time() - starttime
        timedic.update({e: endepoch})
        if e % args.save_rate == args.save_rate - 1:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            for agent in agents:
                agent.save_model(args.save_dir)
        logger.info("episode:{}/{}, average travel time:{}".format(e, args.episodes, env.eng.get_average_travel_time()))
        for agent_id, agent in enumerate(agents):
            logger.info(
                "agent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))
    with open(os.path.join(args.log_dir, '{}.txt'.format(args.config_file[-7:-4])), 'a') as fs:
        fs.write(str(timedic) + '\n')

def test():
    obs = env.reset()
    for agent in agents:
        agent.load_model(args.save_dir, 0)
    for i in range(args.steps):
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(obs[agent_id]))
        obs, rewards, dones, info = env.step(actions)
        #print(rewards)
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
                agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
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
    starttime = time.time()
    with open(os.path.join(args.log_dir, '{}.txt'.format(args.config_file[-7:-4])), 'w') as f:
        f.writelines(str(0) + '\n')
    train(args, env)
    test()
    endtime = time.time() - starttime
    with open(os.path.join(args.log_dir, '{}.txt'.format(args.config_file[-7:-4])), 'a') as f:
        f.writelines(str(endtime))
