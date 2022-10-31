import random
import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from common.registry import Registry


@Registry.register_trainer("base")
class BaseTrainer(ABC):
    '''
    Register BaseTrainer for the whole training and testing process.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="base"
    ):
        self.path = os.path.join('configs/sim', Registry.mapping['command_mapping']['setting'].param['network'] + '.cfg')
        self.seed = Registry.mapping['command_mapping']['setting'].param['seed']
        self.logger = logger
        #self.debug = args['debug']
        self.name = name
        self.cpu = cpu
        self.epoch = 0
        self.step = 0
        self.metric = None
        self.env = None
        self.world = None
        self.agents = None

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        self.load()
        self.create()

    def load(self):
        '''
        load
        Set random seed.

        :param: None
        :return: None
        '''
        self.load_seed_from_config()
        # self.load_logger()
    
    def load_seed_from_config(self):
        '''
        load_seed_from_config
        Set random seed from self.seed.

        :param: None
        :return: None
        '''
        # https://pytorch.org/docs/stable/notes/randomness.html
        if self.seed is None:
            return

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self, logger):
        '''
        load_logger
        Load logger.

        :param: None
        :return: None
        '''
        # TODO: implement logger classes
        self.logger = logger
        if not self.is_debug:
            assert (
                self.config["logger"] is not None
            ), "Specify logger in configs"

            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"

    def create(self):
        '''
        create
        Create world, agents, metric and environment before training process.

        :param: None
        :return: None
        '''
        self.create_world()
        self.create_agents()
        self.create_metrics()
        self.create_env()

    @abstractmethod
    def create_world(self):
        """Derived classes should implement this function."""
    
    @abstractmethod
    def create_agents(self):
        """Derived classes should implement this function."""

    @abstractmethod
    def create_metrics(self):
        """Derived classes should implement this function."""

    def create_env(self):
        """Derived classes should implement this function."""

    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""

    @abstractmethod
    def train_test(self, e):
        """Derived classes should implement this function."""

    @abstractmethod
    def test(self, drop_load=False):
        """Derived classes should implement this function."""
        

