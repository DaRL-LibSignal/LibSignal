import os
import numpy as np
from common.registry import Registry
import math


def max_common_divisor(num):
    minimum = max(num)
    for i in num:
        minimum = math.gcd(int(i), int(minimum))
    return int(minimum)

def tsc_train(trainer):
    '''
    train
    Train the agent(s).
    :param: None
    :return: None
    '''
    total_decision_num = 0
    flush = 0
    for e in range(trainer.episodes):
        # TODO: check this reset agent
        trainer.metric.clear()
        last_obs = trainer.env.reset()  # agent * [sub_agent, feature]
        for a in trainer.agents:
            a.reset()
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if trainer.save_replay and e % trainer.save_rate == 0:
                trainer.env.eng.set_save_replay(True)
                trainer.env.eng.set_replay_file(os.path.join(trainer.replay_file_dir, f"episode_{e}.txt"))
            else:
                trainer.env.eng.set_save_replay(False)
        episode_loss = []
        i = 0
        while i < trainer.steps:
            if i % trainer.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in trainer.agents])  # [agent, intersections]
                if total_decision_num > trainer.learning_start:
                    actions = []
                    for idx, ag in enumerate(trainer.agents):
                        actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))                            
                    actions = np.stack(actions)  # [agent, intersections]
                else:
                    actions = np.stack([ag.sample() for ag in trainer.agents])
                actions_prob = []
                for idx, ag in enumerate(trainer.agents):
                    actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))
                rewards_list = []
                for _ in range(trainer.action_interval):
                    obs, rewards, dones, _ = trainer.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                trainer.metric.update(rewards)
                cur_phase = np.stack([ag.get_phase() for ag in trainer.agents])
                for idx, ag in enumerate(trainer.agents):
                    ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                        obs[idx], cur_phase[idx], dones[idx], f'{e}_{i//trainer.action_interval}_{ag.id}')
                flush += 1
                if flush == trainer.buffer_size - 1:
                    flush = 0
                    # trainer.dataset.flush([ag.replay_buffer for ag in trainer.agents])
                total_decision_num += 1
                last_obs = obs
            if total_decision_num > trainer.learning_start and\
                    total_decision_num % trainer.update_model_rate == trainer.update_model_rate - 1:
                cur_loss_q = np.stack([ag.train() for ag in trainer.agents])  # TODO: training
                episode_loss.append(cur_loss_q)
            if total_decision_num > trainer.learning_start and \
                    total_decision_num % trainer.update_target_rate == trainer.update_target_rate - 1:
                [ag.update_target_network() for ag in trainer.agents]
            if all(dones):
                break
        if len(episode_loss) > 0:
            mean_loss = np.mean(np.array(episode_loss))
        else:
            mean_loss = 0
        
        trainer.writeLog("TRAIN", e, trainer.metric.real_average_travel_time(),\
            mean_loss, trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), trainer.metric.throughput())
        trainer.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, trainer.steps,\
            mean_loss, trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), int(trainer.metric.throughput())))
        if e % trainer.save_rate == 0:
            [ag.save_model(e=e) for ag in trainer.agents]
        trainer.logger.info("episode:{}/{}, real avg travel time:{}".format(e, trainer.episodes, trainer.metric.real_average_travel_time()))
        for j in range(len(trainer.world.intersections)):
            trainer.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, trainer.metric.lane_rewards()[j],\
                 trainer.metric.lane_queue()[j]))
        if trainer.test_when_train:
            trainer.train_test(e)
    # trainer.dataset.flush([ag.replay_buffer for ag in trainer.agents])
    [ag.save_model(e=trainer.episodes) for ag in trainer.agents]

def tsc_train_test(trainer, e):
    '''
    train_test
    Evaluate model performance after each episode training process.
    :param e: number of episode
    :return trainer.metric.real_average_travel_time: travel time of vehicles
    '''
    obs = trainer.env.reset()
    trainer.metric.clear()
    for a in trainer.agents:
        a.reset()
    for i in range(trainer.test_steps):
        if i % trainer.action_interval == 0:
            phases = np.stack([ag.get_phase() for ag in trainer.agents])
            actions = []
            for idx, ag in enumerate(trainer.agents):
                actions.append(ag.get_action(obs[idx], phases[idx], test=True))
            actions = np.stack(actions)
            rewards_list = []
            for _ in range(trainer.action_interval):
                obs, rewards, dones, _ = trainer.env.step(actions.flatten())  # make sure action is [intersection]
                i += 1
                rewards_list.append(np.stack(rewards))
            rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
            trainer.metric.update(rewards)
        if all(dones):
            break
    trainer.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
        e, trainer.episodes, trainer.metric.real_average_travel_time(), trainer.metric.rewards(),\
        trainer.metric.queue(), trainer.metric.delay(), int(trainer.metric.throughput())))
    trainer.writeLog("TEST", e, trainer.metric.real_average_travel_time(),\
        100, trainer.metric.rewards(),trainer.metric.queue(),trainer.metric.delay(), trainer.metric.throughput())
    return trainer.metric.real_average_travel_time()

def tsc_test(trainer, drop_load=True):
    '''
    test
    Test process. Evaluate model performance.
    :param drop_load: decide whether to load pretrained model's parameters
    :return trainer.metric: including queue length, throughput, delay and travel time
    '''
    if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
        if trainer.save_replay:
            trainer.env.eng.set_save_replay(True)
            trainer.env.eng.set_replay_file(os.path.join(trainer.replay_file_dir, f"final.txt"))
        else:
            trainer.env.eng.set_save_replay(False)
    trainer.metric.clear()
    if not drop_load:
        [ag.load_model(trainer.episodes) for ag in trainer.agents]
    attention_mat_list = []
    obs = trainer.env.reset()
    for a in trainer.agents:
        a.reset()
    for i in range(trainer.test_steps):
        if i % trainer.action_interval == 0:
            phases = np.stack([ag.get_phase() for ag in trainer.agents])
            actions = []
            for idx, ag in enumerate(trainer.agents):
                actions.append(ag.get_action(obs[idx], phases[idx], test=True))
            actions = np.stack(actions)
            rewards_list = []
            for j in range(trainer.action_interval):
                obs, rewards, dones, _ = trainer.env.step(actions.flatten())
                i += 1
                rewards_list.append(np.stack(rewards))
            rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
            trainer.metric.update(rewards)
        if all(dones):
            break
    trainer.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (trainer.metric.real_average_travel_time(), \
        trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), trainer.metric.throughput()))
    return trainer.metric

def tscfx_train(trainer):
    '''
    Train the agent(s).
    tscfx agent need experience is added after each duration exausted
    :param: None
    :return: None
    '''
    total_decision_num = 0
    flush = 0
    for e in range(trainer.episodes):
        # TODO: check this reset agent
        trainer.metric.clear()
        last_obs = trainer.env.reset()  # agent * [sub_agent, feature]
        for a in trainer.agents:
            a.reset()
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if trainer.save_replay and e % trainer.save_rate == 0:
                trainer.env.eng.set_save_replay(True)
                trainer.env.eng.set_replay_file(os.path.join(trainer.replay_file_dir, f"episode_{e}.txt"))
            else:
                trainer.env.eng.set_save_replay(False)
        episode_loss = []
        i = 0
        while i < trainer.steps:
            if i % trainer.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in trainer.agents])  # [agent, intersections]
                if total_decision_num > trainer.learning_start:
                    actions = []
                    for idx, ag in enumerate(trainer.agents):
                        actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))                            
                    actions = np.stack(actions)  # [agent, intersections]
                else:
                    actions = np.stack([ag.sample() for ag in trainer.agents])    
                actions_prob = []
                for idx, ag in enumerate(trainer.agents):
                    actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))
                rewards_list = []
                for _ in range(trainer.action_interval):
                    obs, rewards, dones, _ = trainer.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                print([f'{a:2d}' for a in actions])
                print([f'{ag.duration_cur:2d}' for ag in trainer.agents])
                print([f'{ag.duration_residual:2d}' for ag in trainer.agents])             
                print() 
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                trainer.metric.update(rewards)
                cur_phase = np.stack([ag.get_phase() for ag in trainer.agents])
                for idx, ag in enumerate(trainer.agents):
                    # TODO: revise replay buffer for tscfx laster
                    # ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                    #     obs[idx], cur_phase[idx], dones[idx], f'{e}_{i//trainer.action_interval}_{ag.id}')
                    pass
                flush += 1
                if flush == trainer.buffer_size - 1:
                    flush = 0
                    # trainer.dataset.flush([ag.replay_buffer for ag in trainer.agents])
                total_decision_num += 1
                last_obs = obs
            if total_decision_num > trainer.learning_start and\
                    total_decision_num % trainer.update_model_rate == trainer.update_model_rate - 1:
                cur_loss_q = np.stack([ag.train() for ag in trainer.agents])  # TODO: training
                episode_loss.append(cur_loss_q)
            if total_decision_num > trainer.learning_start and \
                    total_decision_num % trainer.update_target_rate == trainer.update_target_rate - 1:
                [ag.update_target_network() for ag in trainer.agents]
            if all(dones):
                break
        if len(episode_loss) > 0:
            mean_loss = np.mean(np.array(episode_loss))
        else:
            mean_loss = 0
        
        trainer.writeLog("TRAIN", e, trainer.metric.real_average_travel_time(),\
            mean_loss, trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), trainer.metric.throughput())
        trainer.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, trainer.steps,\
            mean_loss, trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), int(trainer.metric.throughput())))
        if e % trainer.save_rate == 0:
            [ag.save_model(e=e) for ag in trainer.agents]
        trainer.logger.info("episode:{}/{}, real avg travel time:{}".format(e, trainer.episodes, trainer.metric.real_average_travel_time()))
        for j in range(len(trainer.world.intersections)):
            trainer.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, trainer.metric.lane_rewards()[j],\
                 trainer.metric.lane_queue()[j]))
        if trainer.test_when_train:
            trainer.train_test(e)
    # trainer.dataset.flush([ag.replay_buffer for ag in trainer.agents])
    [ag.save_model(e=trainer.episodes) for ag in trainer.agents]

def tscfx_train_test(trainer, e):
    '''
    train_test
    Evaluate model performance after each episode training process.
    :param e: number of episode
    :return trainer.metric.real_average_travel_time: travel time of vehicles
    '''
    obs = trainer.env.reset()
    trainer.metric.clear()
    for a in trainer.agents:
        a.reset()
    for i in range(trainer.test_steps):
        if i % trainer.action_interval == 0:
            phases = np.stack([ag.get_phase() for ag in trainer.agents])
            actions = []
            for idx, ag in enumerate(trainer.agents):
                actions.append(ag.get_action(obs[idx], phases[idx], test=True))
            actions = np.stack(actions)
            rewards_list = []
            for _ in range(trainer.action_interval):
                obs, rewards, dones, _ = trainer.env.step(actions.flatten())  # make sure action is [intersection]
                i += 1
                rewards_list.append(np.stack(rewards))
            rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
            trainer.metric.update(rewards)
        if all(dones):
            break
    trainer.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
        e, trainer.episodes, trainer.metric.real_average_travel_time(), trainer.metric.rewards(),\
        trainer.metric.queue(), trainer.metric.delay(), int(trainer.metric.throughput())))
    trainer.writeLog("TEST", e, trainer.metric.real_average_travel_time(),\
        100, trainer.metric.rewards(),trainer.metric.queue(),trainer.metric.delay(), trainer.metric.throughput())
    return trainer.metric.real_average_travel_time()

def tscfx_test(trainer, drop_load=True):
    '''
    test
    Test process. Evaluate model performance.
    :param drop_load: decide whether to load pretrained model's parameters
    :return trainer.metric: including queue length, throughput, delay and travel time
    '''
    if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
        if trainer.save_replay:
            trainer.env.eng.set_save_replay(True)
            trainer.env.eng.set_replay_file(os.path.join(trainer.replay_file_dir, f"final.txt"))
        else:
            trainer.env.eng.set_save_replay(False)
    trainer.metric.clear()
    if not drop_load:
        [ag.load_model(trainer.episodes) for ag in trainer.agents]
    attention_mat_list = []
    obs = trainer.env.reset()
    for a in trainer.agents:
        a.reset()
    for i in range(trainer.test_steps):
        if i % trainer.action_interval == 0:
            phases = np.stack([ag.get_phase() for ag in trainer.agents])
            actions = []
            for idx, ag in enumerate(trainer.agents):
                actions.append(ag.get_action(obs[idx], phases[idx], test=True))
            actions = np.stack(actions)
            rewards_list = []
            for j in range(trainer.action_interval):
                obs, rewards, dones, _ = trainer.env.step(actions.flatten())
                i += 1
                rewards_list.append(np.stack(rewards))
            rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
            trainer.metric.update(rewards)
        if all(dones):
            break
    trainer.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (trainer.metric.real_average_travel_time(), \
        trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), trainer.metric.throughput()))
    return trainer.metric