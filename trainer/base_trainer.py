import datetime
import errno
import json
import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

import common
from common.registry import registry
from common.utils import (
    TravelTimeMetric,
)

from environment import TSCEnv
from world import World

@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(
        self,
        task,
        model,
        traffic_env,
        cityflow_settings,
        optim,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        gpu=0,
        cpu=False,
        name="base_trainer"
    ):
        self.name = name
        self.cpu = cpu
        self.epoch = 0
        self.step = 0

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True 
        
        if run_dir is None:
            run_dir = os.getcwd()
        
        if timestamp_id is None:
            timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
                self.device
            )
            timestamp = datetime.datetime.fromtimestamp(
                timestamp.int()
            ).strftime("%Y-%m-%d-%H-%M-%S")
            if identifier:
                self.timestamp_id = f"{timestamp}-{identifier}"
            else:
                self.timestamp_id = timestamp
        else:
            self.timestamp_id = timestamp_id

        try:
            commit_hash = (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        common.__path__[0],
                        "describe",
                        "--always",
                    ]
                )
                .strip()
                .decode("ascii")
            )
        # catch instances where code is not being run from a git repo
        except Exception:
            commit_hash = None

        logger_name = logger if isinstance(logger, str) else logger["name"]
        
        self.config = {
            "task": task,
            "model": model.pop("name"),
            "traffic_env": traffic_env,
            "model_attributes": model,
            "cityflow_settings": cityflow_settings,
            "optim": optim,
            "logger": logger,
            "gpu": 0,
            # "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", self.timestamp_id
                ),
                "results_dir": os.path.join(
                    run_dir, "results", self.timestamp_id
                ),
                "logs_dir": os.path.join(
                    run_dir, "logs", logger_name, self.timestamp_id
                ),
            },
        }

        # self.config["dataset"] = dataset

        if not is_debug:
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)
        
        self.is_debug = is_debug
        # print(yaml.dump(self.config, default_flow_style=False))
        self.load()
        self.create()


    def load(self):
        self.load_seed_from_config()
        self.load_logger()
    
    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if not self.is_debug:
            assert (
                self.config["logger"] is not None
            ), "Specify logger in config"

            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"

            self.logger = registry.get_logger_class(logger_name)(self.config)

    def create(self):
        self.create_world()
        self.create_agents()
        self.create_metric()
        self.create_env()

    def create_world(self):
        self.world = World('/home/lxl/LibSignalFrame/cityflow.cfg', thread_num=self.config['optim']['thread'])
        # self.world = World(self.config['cityflow_settings'], thread_num=self.config['optim']['thread'])
    
    @abstractmethod
    def create_agents(self):
        """Derived classes should implement this function."""

    def create_metric(self):
        self.metric = TravelTimeMetric(self.world)

    def create_env(self):
        self.env = TSCEnv(self.world, self.agents, self.metric)

    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""

    @abstractmethod
    def train_test(self):
        """Derived classes should implement this function."""

    @abstractmethod
    def test(self):
        """Derived classes should implement this function."""
        

