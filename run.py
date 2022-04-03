import os.path

from environment import TSCEnv
from world import World
from task.traffic_light import TrafficLightDQN
from metric import TravelTimeMetric
import argparse
import logging
from datetime import datetime
from common.utils import *
from common.registry import Registry
from common import interface
from agent import colight
from common.onpolicy_dataset import OnPolicyDataset

# parseargs
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--thread', type=int, default=4, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card

parser.add_argument('-t', '--task', type=str, default="tc", help="task type to run")
parser.add_argument('-m', '--model', type=str, default="colight", help="model type of agent in RL environment")
parser.add_argument('-p', '--prefix', type=str, default='2', help="the number of predix in this running process")

parser.add_argument('--mask_type', type=int, default=0, help='used to specify the type of softmax')
parser.add_argument('--test_when_train', action="store_false", default=True)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

if __name__ == '__main__':

    test_when_train = args.test_when_train

    config_path = r'.\config\cityflow4X4.cfg'
    config = json.load(open(config_path, "r"))
    roadnet_config_path = os.path.join('data', config['roadnetFile'])
    interface.Graph_World_Interface(roadnet_config_path)

    # initiate logger path to keep track of this running process
    interface.Logger_path_Interface(args.task, args.model, args.prefix)
    if not os.path.exists(Registry.mapping['logger_mapping']['output_path'].path):
        os.makedirs(Registry.mapping['logger_mapping']['output_path'].path)
    pLogger = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'logger')
    pModel = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model')
    pDataset = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'dataset')

    # create world
    world = World(config_path, thread_num=args.thread)

    # TODO:update the below dict (already done)
    metric = TravelTimeMetric(world)

    # initiate logger dictionary
    logger_path = r'.\config\log.cfg'
    interface.Logger_param_Interface(logger_path)

    logger = logging.getLogger('run')
    logger.setLevel(logging.DEBUG)
    prefix = "CoLight_" + datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists(pLogger):
        os.makedirs(pLogger)
    fh = logging.FileHandler(os.path.join(pLogger, f'{prefix}.log'))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # initiate traffic dictionary
    traffic_path = r'.\config\traffic.cfg'
    interface.Traffic_path_Interface(traffic_path)

    # initiate training config
    task_path = r'.\config\training.cfg'
    interface.Task_param_Interface(task_path)

    # initiate model object and model dictionary
    model_path = r'.\config\model.cfg'
    interface.ModelAgent_param_Interface(model_path)
    agent = Registry.mapping['model_mapping']['model'](world, prefix)

    # create env
    env = TSCEnv(world, [agent], metric)

    # initiate dataset path
    dataset = Registry.mapping['dataset_mapping']['dataset'](pDataset)

    player = TrafficLightDQN(agent, env, world, dataset, logger, prefix, test_when_train)
    player.train()
    player.test()
