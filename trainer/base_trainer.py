import random
from abc import ABC, abstractmethod
import numpy as np
import torch
from common.registry import Registry


@Registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(
        self,
        args,
        logger,
        gpu=0,
        cpu=False,
        name="base"
    ):
        self.args = args
        self.cityflow_path = args['cityflow_path']
        self.seed = args['seed']
        self.logger = logger
        self.debug = args['debug']
        self.name = name
        self.cpu = cpu
        self.epoch = 0
        self.step = 0
        self.metric = None
        self.env = None
        self.world = None
        self.agents = None
        self.metric = None

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        self.load()
        self.create()

    def load(self):
        self.load_seed_from_config()
        # self.load_logger()
    
    def load_seed_from_config(self):
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
        self.create_world()
        self.create_agents()
        self.create_metric()
        self.create_env()

    @abstractmethod
    def create_world(self):
        """Derived classes should implement this function."""
    
    @abstractmethod
    def create_agents(self):
        """Derived classes should implement this function."""

    @abstractmethod
    def create_metric(self):
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
        

