import logging
import os
import numpy as np
import gym
from generator.lane_vehicle import LaneVehicleGenerator
from common.registry import registry
from trainers.base_trainer import BaseTrainer
from agents.frap_agent import FRAP_DQNAgent

@registry.register_trainer("tsc_ng")
class TSCNGTrainer(BaseTrainer):
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
        name="tsc_ng"
    ):
        super().__init__(
            task=task,
            model=model,
            traffic_env=traffic_env,
            cityflow_settings=cityflow_settings,
            optim=optim,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.optim=optim

    
    def create_agents(self):
        self.agents = []
        for i in self.world.intersections:
            action_space = gym.spaces.Discrete(len(i.phases))
            # TODO how to divide process of creating different agents
            self.agents.append(FRAP_DQNAgent(
                action_space,
                LaneVehicleGenerator(
                    self.world, i, ["lane_count"], in_only=True, average=None),
                LaneVehicleGenerator(
                    self.world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
                self.world,
                self.config,
                i.id
            ))


    def train(self):
        for e in range(self.optim['episodes']):
            last_obs = self.env.reset()
            episodes_rewards = [0 for i in self.agents]
            episodes_decision_num = 0
            i = 0
            while i < self.optim['steps']:
                if i % self.optim['action_interval'] == 0:
                    actions = []
                    for agent_id, agent in enumerate(self.agents):
                        actions.append(agent.get_action(last_obs[agent_id]))

                    rewards_list = []
                    for _ in range(self.optim['action_interval']):
                        obs, rewards, dones, _ = self.env.step(actions)
                        i += 1
                        rewards_list.append(rewards)
                    rewards = np.mean(rewards_list, axis=0)

                    for agent_id, agent in enumerate(self.agents):
                        agent.remember(
                            last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                    last_obs = obs

                total_time = i + e * self.optim['steps']
                for agent_id, agent in enumerate(self.agents):
                    if total_time > agent.learning_start and total_time % agent.update_model_freq == 0:
                        agent.replay()
                    if total_time > agent.learning_start and total_time % agent.update_target_model_freq == 0:
                        agent.update_target_network()
                if all(dones):
                    break

            if e % self.optim['save_rate'] == 0:
                if not os.path.exists(self.optim['savemodel_dir']):
                    os.makedirs(self.optim['savemodel_dir'])
                for agent in self.agents:
                    agent.save_model(self.optim['savemodel_dir'])
            logging.info("episode:{}, average travel time:{}".format(
                e, self.env.eng.get_average_travel_time()))
            for agent_id, agent in enumerate(self.agents):
                logging.info("agent:{}, mean_episode_reward:{}".format(
                    agent_id, episodes_rewards[agent_id] / episodes_decision_num))

    def train_test(self):
        pass

    def test(self):
        obs = self.env.reset()
        for agent in self.agents:
            agent.load_model(self.optim['savemodel_dir'])
            agent.if_test = 1
        for i in range(self.optim['steps']):
            if i % self.optim['action_interval'] == 0:
                actions = []
                for agent_id, agent in enumerate(self.agents):
                    actions.append(agent.get_action(obs[agent_id]))
            obs, rewards, dones, info = self.env.step(actions)
            logging.info("episode:{}, average travel time:{}".format(
                i, self.env.eng.get_average_travel_time()))
            if all(dones):
                break
        logging.info(self.env.eng.get_average_travel_time())