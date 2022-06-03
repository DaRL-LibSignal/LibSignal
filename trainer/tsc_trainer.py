import os
import numpy as np
from common.metric import TravelTimeMetric
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    def __init__(
        self,
        args,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            args=args,
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['trainer_setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['trainer_setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['trainer_setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['trainer_setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['logger_setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['trainer_setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['trainer_setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['trainer_setting'].param['update_target_rate']
        self.replay_file_dir = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                            Registry.mapping['logger_mapping']['logger_setting'].param['replay_dir'])
        # TODO: pass in dataset
        self.prefix = self.args['prefix']
        self.dataset = Registry.mapping['dataset_mapping'][self.args['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                         Registry.mapping['logger_mapping']['logger_setting'].param['data_dir'])
        )
        # self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.test_when_train = self.args['test_when_train']
        self.yellow_time = self.world.intersections[0].yellow_phase_time
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                     'logger', self.logger.name + '_details.log')

    def create_world(self):
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][self.args['world']](
            self.path, Registry.mapping['world_mapping']['traffic_setting'].param['thread_num'])

    def create_metric(self):
        self.metric = TravelTimeMetric(self.world)

    def create_agents(self):
        self.agents = []
        # for i in self.world.intersections:
        #     self.agents.append(Registry.mapping['model_mapping'][self.args['agent']](self.world, i.id))
        agent = Registry.mapping['model_mapping'][self.args['agent']](self.world, 0)
        if Registry.mapping['model_mapping']['model_setting'].param['name'] != 'maddpg' and\
                Registry.mapping['model_mapping']['model_setting'].param['name'] != "maddpg_v2":
            print(agent.model)

        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][self.args['agent']](self.world, i))

        if Registry.mapping['model_mapping']['model_setting'].param['name'] == 'maddpg':
            for ag in self.agents:
                ag.link_agents(self.agents)
            print(ag.q_model)
            print(ag.p_model)
        if Registry.mapping['model_mapping']['model_setting'].param['name'] == 'maddpg_v2':
            print(self.agents[0].agents[0].actor)
            print(self.agents[0].agents[0].critic)

    def create_env(self):
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        total_decision_num = 0
        flush = 0
        for e in range(self.episodes):
            # TODO: check this reset agent
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            if e % self.save_rate == 0:
                # self.env.eng.set_save_replay(True)
                if not os.path.exists(self.replay_file_dir):
                    os.makedirs(self.replay_file_dir)
                # self.env.eng.set_replay_file(self.replay_file_dir + f"/episode_{e}.txt")  # TODO: replay here
            else:
                pass
                # self.env.eng.set_save_replay(False)
            episodes_rewards = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
            episodes_queue = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
            episodes_delay = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
            episodes_throughput = 0
            episodes_decision_num = 0
            episode_loss = []
            i = 0

            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])


                    # TODO: temporary for test maddpg agent
                    if Registry.mapping['model_mapping']['model_setting'].param['name'] == 'maddpg' or\
                            Registry.mapping['model_mapping']['model_setting'].param['name'] == "maddpg_v2":
                        actions_prob = []
                        for idx, ag in enumerate(self.agents):
                            actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    reward_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        reward_list.append(np.stack(rewards))
                    episodes_queue += (np.stack(np.array([ag.get_queue() for ag in self.agents], dtype=np.float32))).flatten() # [intersections,]
                    episodes_delay += (np.stack(np.array([ag.get_delay() for ag in self.agents], dtype=np.float32))).flatten() # [intersections,]
                    rewards = np.mean(reward_list, axis=0)  # [agent, intersection]
                    episodes_rewards += rewards.flatten()

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    # TODO: construct database here
                    for idx, ag in enumerate(self.agents):
                        # TODO: test for PFRL
                        if Registry.mapping['model_mapping']['model_setting'].param['name'] == 'ppo_pfrl':
                            ag.do_observe(obs[idx], cur_phase[idx], rewards[idx], dones[idx])
                        # TODO: temporary for test maddpg agent
                        elif Registry.mapping['model_mapping']['model_setting'].param['name'] == 'maddpg' or\
                            Registry.mapping['model_mapping']['model_setting'].param['name'] == 'maddpg_v2':
                            ag.remember(last_obs[idx], last_phase[idx], actions_prob[idx], rewards[idx],
                                        obs[idx], cur_phase[idx], f'{e}_{i // self.action_interval}_{ag.id}')
                        else:
                            ag.remember(last_obs[idx], last_phase[idx], actions[idx], rewards[idx],
                                        obs[idx], cur_phase[idx], f'{e}_{i//self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])

                    episodes_decision_num += 1
                    total_decision_num += 1
                    last_obs = obs
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:

                    cur_loss_q = np.stack([ag.train() for ag in self.agents])  # TODO: train here

                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            mean_queue = np.sum(episodes_queue) / (episodes_decision_num * len(self.world.intersections))
            mean_delay = np.sum(episodes_delay) / (episodes_decision_num * len(self.world.intersections))
            episodes_throughput = self.env.world.get_cur_throughput()
            mean_reward = np.sum(episodes_rewards) / episodes_decision_num
            cur_travel_time = self.env.world.get_average_travel_time()
            # sumo env has 2 travel time: [real travel time, planned travel time(aligned with Cityflow)]
            self.writeLog("TRAIN", e, cur_travel_time[0], cur_travel_time[1], mean_loss, mean_reward, mean_queue, mean_delay, episodes_throughput)
            self.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps,
                                                        mean_loss, mean_reward, mean_queue, mean_delay, int(episodes_throughput)))
            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]
            self.logger.info("episode:{}/{}, real avg travel time:{}, planned avg travel time:{}".format(e, self.episodes, cur_travel_time[0], cur_travel_time[1]))
            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, episodes_rewards[j] / episodes_decision_num, episodes_queue[j]/episodes_decision_num, episodes_delay[j]/episodes_decision_num))
            if self.test_when_train:
                self.train_test(e)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        [ag.save_model(e=self.episodes) for ag in self.agents]

    def train_test(self, e):
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        ep_rwds = [0 for _ in range(len(self.world.intersections))]
        ep_queue = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        ep_delay = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        ep_throughput = 0
        eps_nums = 0
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                ep_queue += (np.stack(np.array([ag.get_queue() for ag in self.agents], dtype=np.float32))).flatten() # avg_queue of intersections
                ep_delay += (np.stack(np.array([ag.get_delay() for ag in self.agents], dtype=np.float32))).flatten() # avg_queue of intersections
                rewards = np.mean(rewards_list, axis=0)
                ep_rwds += rewards.flatten()
                eps_nums += 1
            if all(dones):
                break
        mean_rwd = np.sum(ep_rwds) / eps_nums
        mean_queue = np.sum(ep_queue) / (eps_nums * len(self.world.intersections))
        mean_delay = np.sum(ep_delay) / (eps_nums * len(self.world.intersections))
        ep_throughput = self.env.world.get_cur_throughput()
        trv_time = self.env.world.get_average_travel_time()
        self.logger.info("Test step:{}/{}, real travel time :{}, planned travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(e, self.steps, trv_time[0], trv_time[1], mean_rwd,mean_queue, mean_delay, int(ep_throughput)))
        self.writeLog("TEST", e, trv_time[0], trv_time[1], 100, mean_rwd,mean_queue,mean_delay, ep_throughput)
        return trv_time

    def test(self, drop_load=True):
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]
        attention_mat_list = []
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        ep_rwds = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        ep_queue = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        ep_delay = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        ep_throughput = 0
        eps_nums = 0
        # lane_delay = np.array([value for _, value in self.env.world.get_lane_delay().items()]).mean()
        # lane_queue_length = np.array([value for _, value in self.env.world.get_lane_queue_length().items()]).mean()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                ep_queue += (np.stack(np.array([ag.get_queue() for ag in self.agents], dtype=np.float32))).flatten()
                ep_delay += (np.stack(np.array([ag.get_delay() for ag in self.agents], dtype=np.float32))).flatten()
                rewards = np.mean(rewards_list, axis=0)
                ep_rwds += rewards.flatten()
                eps_nums += 1
                # lane_delay += np.array([value for _, value in self.env.world.get_lane_delay().items()]).mean()
                # lane_queue_length += np.array(
                #     [value for _, value in self.env.world.get_lane_queue_length().items()]).mean()
            if all(dones):
                break
        mean_rwd = np.sum(ep_rwds) / eps_nums
        trv_time = self.env.world.get_average_travel_time()
        # lane_delay = lane_delay / self.episodes
        # lane_queue_length = lane_queue_length / self.episodes
        # self.logger.info("Final Travel Time is %.4f, and mean rewards %.4f" % (trv_time, mean_rwd))
        # self.logger.info("Final average lane delay is %.4f." % lane_delay)
        # self.logger.info("Final lane length is %.4f." % lane_queue_length)
        mean_queue = np.sum(ep_queue) / (eps_nums * len(self.world.intersections))
        mean_delay = np.sum(ep_delay) / (eps_nums * len(self.world.intersections))
        ep_throughput = self.env.world.get_cur_throughput()
        self.logger.info("Final Travel Time is %.4f, Planned Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (trv_time[0], trv_time[1], mean_rwd, mean_queue, mean_delay, ep_throughput))
        
        # TODO: add attention record
        if Registry.mapping['logger_mapping']['logger_setting'].param['get_attention']:
            pass
        # self.env.eng.set_save_replay(True)
        # self.env.eng.set_replay_file(self.replay_file_dir + "replay.txt")
        return trv_time

    def writeLog(self, mode, step, travel_time, planned_tt, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        """
        :param mode: "TRAIN" OR "TEST"
        :param step: int
        """
        res = self.args['model']['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % planned_tt + '\t' + "%.1f" % loss + "\t" + "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

