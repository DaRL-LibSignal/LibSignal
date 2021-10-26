import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from generator import IntersectionVehicleGenerator
from agent.intellilight_agent import IntelliLightAgent, paras
from metric import TravelTimeMetric
import argparse
import os
import json
import copy
import math
import time
import numpy as np
from tensorflow import set_random_seed
import random

SEED = 31200
random.seed(SEED)
np.random.seed(SEED)
set_random_seed((SEED))

"""
currently only support training on single agent since its designed as a single intersection algorithm.
you may test it in the multi-agent with parameter-sharing.
"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='path of config file')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument("-s", "--silent", action="store_true")
    return parser.parse_args()
args = parse_arguments()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for idx, i in enumerate(world.intersections):
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(IntelliLightAgent(
        action_space,
        [
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="lane"),
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average="lane"),
            LaneVehicleGenerator(world, i, ["lane_waiting_time_count"], in_only=True, average="lane"),
            IntersectionVehicleGenerator(world, i, targets=["vehicle_map"])
        ],
        [
            LaneVehicleGenerator(world, i, ["lane_waiting_count", "lane_delay", "lane_waiting_time_count"],
                                 in_only=True, average="all"),
            IntersectionVehicleGenerator(world, i, targets=["passed_count", "passed_time_count"])
        ],
        world,
        idx
    ))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

class TrafficLightDQN:

    def __init__(self, agents, env):
        self.agent = agents[0]
        self.env = env
        self.world = world
        self.yellow_time = self.world.intersections[0].yellow_phase_time

    def _generate_pre_train_ratios(self, phase_min_time, em_phase):
        phase_traffic_ratios = [phase_min_time]

        # generate how many varients for each phase
        for i, phase_time in enumerate(phase_min_time):
            if i == em_phase:
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            else:
                # pass
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            for j in range(5, 20, 5):
                gen_phase_time = copy.deepcopy(phase_min_time)
                gen_phase_time[i] += j
                phase_traffic_ratios.append(gen_phase_time)

        return phase_traffic_ratios

    def get_phase(self, action, last_phase):
        if action == 0:
            return last_phase
        else:
            phase = last_phase + 1
            if phase >= 8:
                phase = 0

        return phase


    def train(self, if_pretrain, use_average):

        if if_pretrain:
            total_run_cnt = paras["RUN_COUNTS_PRETRAIN"] #10000
            phase_traffic_ratios = self._generate_pre_train_ratios(paras["BASE_RATIO"], em_phase=0)  # en_phase=0
            pre_train_count_per_ratio = math.ceil(total_run_cnt / len(phase_traffic_ratios))
            ind_phase_time = 0
        else:
            total_run_cnt = paras["RUN_COUNTS"]

        # initialize output streams
        if not os.path.exists(paras["PATH_TO_OUTPUT"]):
            os.makedirs(paras["PATH_TO_OUTPUT"])
        file_name_memory = os.path.join(paras["PATH_TO_OUTPUT"], "memories.txt")

        num_step = args.steps

        current_time = 0  # in seconds

        obs = env.reset()
        ob = obs[0]
        last_action = 0

        total_steps = 0

        while total_steps < total_run_cnt:
            total_steps += 1
            if current_time >= 3600:
                obs = env.reset()
                ob = obs[0]
                last_action = 0
                current_time = 0
            if if_pretrain:
                if current_time > pre_train_count_per_ratio:
                    print("Terminal occured. Episode end.")
                    self.env.reset()
                    ind_phase_time += 1
                    if ind_phase_time >= len(phase_traffic_ratios):
                        break

                    current_time = self.env.eng.get_current_time()  # in seconds

                phase_time_now = phase_traffic_ratios[ind_phase_time]

            f_memory = open(file_name_memory, "a")

            if if_pretrain:
                _, q_values = self.agent.choose(state=ob, count=current_time, if_pretrain=if_pretrain)
                if ob.time_this_phase[0][0] < phase_time_now[ob.cur_phase[0][0]]:
                    action_pred = 0
                else:
                    action_pred = 1
                # print(ob.time_this_phase)
                # print(ob.queue_length)
                action = self.agent.next_phase(last_action) if action_pred else last_action
            else:
                # get action based on e-greedy, combine current state
                action, q_values = self.agent.choose(state=ob, count=current_time, if_pretrain=if_pretrain)

            next_obs, rewards, dones, info = env.step([action])
            if not action == last_action:
                for _ in range(self.yellow_time):
                    next_obs, rewards, dones, info = env.step([action])

            reward = rewards[0]
            next_ob = next_obs[0]
            current_time = self.env.eng.get_current_time()


            # remember
            self.agent.remember(ob, 1 - (action==last_action), reward, next_ob)

            # output to std out and file
            memory_str = 'time = %d\taction = %d\tcurrent_phase = %d\tnext_phase = %d\treward = %f' \
                         '\t%s' \
                         % (current_time, action,
                            ob.cur_phase[0][0],
                            ob.next_phase[0][0],
                            reward, repr(q_values))
            print(memory_str)
            f_memory.write(memory_str + "\n")
            f_memory.close()

            if not if_pretrain:
                # update network
                self.agent.update_network(if_pretrain, use_average, total_steps)
                self.agent.update_network_bar()

            last_action = action
            ob = next_ob


        if if_pretrain:
            self.agent.set_update_outdated()
            self.agent.update_network(if_pretrain, use_average, total_steps)
            self.agent.update_network_bar()
        self.agent.reset_update_count()
        print("END")

def test(env, args, model_name):
    env.agents[0].load_model(model_name)
    i = 0
    obs = env.reset()
    last_action = -1
    while i < args.steps:
        actions = []
        for agent_id, agent in enumerate(env.agents):
            actions.append(agent.get_action(obs[agent_id]))
        action = actions[0]
        obs, rewards, dones, info = env.step(actions)
        i += 1
        if not action == last_action and i != 0:
            for _ in range(env.world.intersections[0].yellow_phase_time):
                next_obs, rewards, dones, info = env.step([action])
                i += 1
        # print(rewards)

        if all(dones):
            break
        last_action = action

    trv_time = env.eng.get_average_travel_time()
    print("Final Travel Time is %.4f" % trv_time)
    return trv_time

if __name__ == "__main__":
    player = TrafficLightDQN(agents, env)
    # player.train(if_pretrain=True, use_average=True)
    # player.train(if_pretrain=False, use_average=False)

    # test(env, args, "602.0")
    test(env, args, "init_model")
