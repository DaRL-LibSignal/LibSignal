import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
import time
from datetime import datetime
import argparse


# parseargs
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--thread_num', type=int, default=4, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card

parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="maxpressure", help="agent type of agents in RL environment")
# parser.add_argument('-w', '--world', type=str, default="cityflow", help="simulator type")
parser.add_argument('-w', '--world', type=str, default="sumo", help="simulator type")
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')
# parser.add_argument('--path', type=str, default='configs/cityflow_cologne1.cfg', help='path to cityflow path')
parser.add_argument('--path', type=str, default='configs/sumohz1x1.cfg', help='path to cityflow path')
parser.add_argument('--prefix', type=str, default='0', help="the number of predix in this running process")
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")

parser.add_argument('--mask_type', type=int, default=0, help='used to specify the type of softmax')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--test_when_train', action="store_false", default=True)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu


class Runner:
    def __init__(self, pArgs):
        self.config = build_config(pArgs)
        self.config_registry()

    def config_registry(self):
        cityflow_setting = json.load(open(self.config['path'], 'r'))

        roadnet_path = os.path.join(cityflow_setting['dir'], cityflow_setting['roadnetFile'])
        if self.config['model'].get('graphic', False):
            interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class
        interface.Logger_path_Interface(self.config['task'], self.config['agent'], self.config['prefix'])
        if not os.path.exists(Registry.mapping['logger_mapping']['output_path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['output_path'].path)
        interface.Logger_param_Interface(self.config['logger'])  # register logger path
        interface.Traffic_param_Interface(self.config['traffic'])
        interface.Trainer_param_Interface(self.config['trainer'])
        interface.ModelAgent_param_Interface(self.config['model'])

    def run(self):
        self.config_registry()
        logger = setup_logging(self.config)
        self.trainer = Registry.mapping['trainer_mapping'][self.config['task']](self.config, logger)
        self.task = Registry.mapping['task_mapping'][self.config['task']](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    test = Runner(args)
    test.run()

