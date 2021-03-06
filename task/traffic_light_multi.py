import os
import numpy as np
from common.registry import Registry


class TrafficLightDQNMulti:
    def __init__(self, agents, env, world, dataset, logging_tool, prefix, test_when_train):
        self.agents = agents
        self.world = world
        self.env = env
        self.prefix = prefix
        self.episodes = Registry.mapping['task_mapping']['task_setting'].param['episodes']
        self.steps = Registry.mapping['task_mapping']['task_setting'].param['steps']
        self.test_steps = Registry.mapping['task_mapping']['task_setting'].param['test_steps']
        self.buffer_size = Registry.mapping['task_mapping']['task_setting'].param['buffer_size']
        self.action_interval = Registry.mapping['task_mapping']['task_setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['logger_setting'].param['save_rate']
        self.learning_start = Registry.mapping['task_mapping']['task_setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['task_mapping']['task_setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['task_mapping']['task_setting'].param['update_target_rate']

        self.dataset = dataset
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.logging_tool = logging_tool
        self.test_when_train = test_when_train
        self.yellow_time = self.world.intersections[0].yellow_phase_time
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                     'logger', self.prefix + '.log')

        log_handle = open(self.log_file, 'w')
        log_handle.close()
        self.replay_file_dir = os.path.join(os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                                         'replay'))
        if not os.path.exists(self.replay_file_dir):
            os.makedirs(self.replay_file_dir)

    def train(self):
        # total_decision_num = 0
        flush = 0
        for e in range(self.episodes):
            last_obs = self.env.reset()  # checked np.array [intersection, feature]
            if e % self.save_rate == 0:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(self.replay_file_dir + f"/episode_{e}.txt")
            else:
                self.env.eng.set_save_replay(False)
            episodes_rewards = np.array([0 for i in range(len(self.world.intersections))], dtype=np.float32)  # checked [intersections,1]
            episodes_decision_num = 0
            episode_loss = []
            # flag=False
            i = 0
            while i < self.steps:
                flag=False
                if i % self.action_interval == 0:
                    actions = []
                    for agent_id, agent in enumerate(self.agents):
                        agent.if_test = 0
                        actions.append(agent.get_action(last_obs[agent_id]))
                
                    reward_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions)  
                        i += 1  # reward: checked np.array [intersection, 1]
                        reward_list.append(rewards)
                    rewards = np.mean(reward_list, axis=0)  
                    # episodes_rewards += rewards  # TODO: check accumulation

                    # cur_phase = self.agent.get_phase()
                    # TODO: construct database here
                    # self.agent.remember(last_obs, last_phase, actions, rewards, obs, cur_phase,
                    #                     f'{e}_{i//self.action_interval}')

                    for agent_id, agent in enumerate(self.agents):
                        agent.remember(
                            last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1


                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush(self.agent.replay_buffer)
                        for agent_id, agent in enumerate(self.agents):
                            self.dataset.flush(agent.replay_buffer)

                    # episodes_decision_num += 1
                    # total_decision_num += 1
                    last_obs = obs

                total_decision_num = i + e * self.steps
                cur_loss_q = []

                for agent_id, agent in enumerate(self.agents):
                    if total_decision_num > self.learning_start and\
                         total_decision_num % self.update_model_rate == 0:
                        flag=True
                        cur_loss_q.append(agent.train())
                    if total_decision_num > self.learning_start and\
                         total_decision_num % self.update_target_rate == 0:
                        agent.update_target_network()
                
                if flag and cur_loss_q[0]!= None:
                    episode_loss.append(cur_loss_q)
                    
                # if total_decision_num > self.learning_start and\
                #         total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                #     cur_loss_q = self.agent.train()  

                #     episode_loss.append(cur_loss_q)
                # if total_decision_num > self.learning_start and \
                #         total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                #     self.agent.update_target_network()

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            cur_travel_time = self.env.eng.get_average_travel_time()
            mean_reward = np.sum(episodes_rewards) / episodes_decision_num
            self.writeLog("TRAIN", e, cur_travel_time, mean_loss, mean_reward)
            self.logging_tool.info(
                "step:{}/{}, q_loss:{}, rewards:{}".format(i, self.episodes,
                                                           mean_loss, mean_reward))
            if e % self.save_rate == 0:
                for agent in self.agents:
                    agent.save_model(e=e)
            self.logging_tool.info(
                "episode:{}/{}, average travel time:{}".format(e, self.episodes, cur_travel_time))
            for agent_id,agent in enumerate(self.agents):
                self.logging_tool.debug(
                    "intersection:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))
            if self.test_when_train:
                self.train_test(e)
        self.dataset.flush(self.agent.queue)
        for agent in self.agents:
            self.dataset.flush(agent.queue)
            agent.save_model(e=self.episodes)

    def train_test(self, e):
        obs = self.env.reset()
        ep_rwds = [0 for _ in range(len(self.world.intersections))]
        eps_nums = 0
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(self.agents):
                    agent.if_test = 1
                    actions.append(agent.get_action(obs[agent_id]))
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                ep_rwds += rewards
                eps_nums += 1
            if all(dones):
                break
        mean_rwd = np.sum(ep_rwds) / eps_nums
        trv_time = self.env.eng.get_average_travel_time()
        # self.logging_tool.info("Final Travel Time is %.4f, and mean rewards %.4f" % (trv_time,mean_rwd))
        self.logging_tool.info(
            "Test step:{}/{}, travel time :{}, rewards:{}".format(e, self.episodes, trv_time, mean_rwd))
        self.writeLog("TEST", e, trv_time, 100, mean_rwd)
        return trv_time

    def test(self, drop_load=False):
        if not drop_load:
            for agent in self.agents:
                agent.load_model(self.episodes)
                agent.if_test = 1
        obs = self.env.reset()
        ep_rwds = [0 for i in range(len(self.world.intersections))]
        eps_nums = 0
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(self.agents):
                    actions.append(agent.get_action(obs[agent_id]))
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                ep_rwds += rewards
                eps_nums += 1
            if all(dones):
                break
        mean_rwd = np.sum(ep_rwds) / eps_nums
        trv_time = self.env.eng.get_average_travel_time()
        self.logging_tool.info("Final Travel Time is %.4f, and mean rewards %.4f" % (trv_time, mean_rwd))
        # # TODO: add attention record
        # if Registry.mapping['logger_mapping']['logger_setting'].param['get_attention']:
        #     pass
        return trv_time

    def writeLog(self, mode, step, travel_time, loss, cur_rwd):
        """
        :param mode: "TRAIN" OR "TEST"
        :param step: int
        """
        res = "Frap" + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" + "%.2f" % cur_rwd
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()
