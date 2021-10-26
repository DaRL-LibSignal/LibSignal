import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import RLAgent
from metric import TravelTimeMetric
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=100, help='number of steps')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(RLAgent(
        action_space,
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average="road"),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True)
    ))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
for i in range(args.steps):
    if i % 5 == 0:
        actions = env.action_space.sample()
    obs, rewards, dones, info = env.step(actions)
    #print(obs[0].shape)
    print(rewards)
    #print(info["metric"])

print("Final Travel Time is %.4f" % env.metric.update(done=True))