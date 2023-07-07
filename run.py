import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse


# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=4, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card
parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process")
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--interface', type=str, default="traci", choices=['libsumo','traci'], help="interface type") # libsumo(fast) or traci(slow)
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx','real'], help="method of calculating delay") # apx(approximate) or real
parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="frap_move", help="agent type of agents in RL environment")
parser.add_argument('-w', '--world', type=str, default="sumo", choices=['cityflow','sumo'], help="simulator type")
parser.add_argument('-n', '--network', type=str, default="sumo7x7", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        """
        self.config, self.duplicate_config = build_config(pArgs)
        self.config_registry()

    def config_registry(self):
        """
        Register config into Registry class
        """

        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)  # register logger path
        interface.World_param_Interface(self.config)
        if self.config['model'].get('graphic', False):
            param = Registry.mapping['world_mapping']['setting'].param
            if self.config['command']['world'] in ['cityflow', 'sumo']:
                roadnet_path = param['dir'] + param['roadnetFile']
            else:
                roadnet_path = param['road_file_addr']
            interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class
        interface.Logger_path_Interface(self.config)
        # make output dir if not exist
        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)        
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self):
        logger = setup_logging(logging_level)
        self.trainer = Registry.mapping['trainer_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](logger)
        self.task = Registry.mapping['task_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    test = Runner(args)
    test.run()

