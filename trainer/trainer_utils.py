import os
import numpy as np
from common.registry import Registry
import math
from copy import deepcopy


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

def tsc_ma_train(trainer):
    '''
    Train the agent(s).
    tscfx agent need experience is added after each duration exausted
    :param: None
    :return: None
    '''
    total_decision_num = 0
    for e in range(trainer.episodes):
        # TODO: check this reset agent
        flag = False
        trainer.env.reset()
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if trainer.save_replay and e % trainer.save_rate == 0:
                trainer.env.eng.set_save_replay(True)
                trainer.env.eng.set_replay_file(os.path.join(trainer.replay_file_dir, f"episode_{e}.txt"))
            else:
                trainer.env.eng.set_save_replay(False)
        i = 0
        last_obs = {agent_id: None for agent_id in trainer.agents}
        last_phases ={ agent_id: None for agent_id in trainer.agents}
        # actions = {agent_id: None for agent_id in trainer.agents}
        acc_rewards = {agent_id: 0 for agent_id in trainer.agents}
        dones = {agent_id: None for agent_id in trainer.agents}
        episode_loss = {agent_id: [] for agent_id in trainer.agents}
        while i < trainer.steps:
            if i % trainer.action_interval == 0:
                # Clear last round actions
                actions = {agent_id: None for agent_id in trainer.agents.keys()}
                actions_prob = {agent_id: None for agent_id in trainer.agents.keys()}

                for agent_id in trainer.env.agent_iter():
                    if trainer.env._agent_selector.is_last():
                        flag = True
                    cur_ob, cur_phase, reward, done, _ = trainer.env.last()
                    acc_rewards[agent_id] += reward
                    if total_decision_num > trainer.learning_start:
                        action = trainer.agents[agent_id].get_action(cur_ob, cur_phase, test=False)
                    else:
                        action = trainer.agents[agent_id].sample()
                    # TODO: make it more general
                    action_prob = trainer.agents[agent_id].get_action_prob(cur_ob, cur_phase)
                    actions[agent_id] = action
                    actions_prob[agent_id] = action_prob

                    assert type(action) == int
                    trainer.env.step(action)

                    if i != 0:
                        trainer.agents[agent_id].remember(last_obs[agent_id], last_phases[agent_id], f_actions[agent_id], f_actions_prob[agent_id], \
                                                        acc_rewards[agent_id]/trainer.action_interval, cur_ob, cur_phase, done, \
                                                        f'{e}_{i//trainer.action_interval-1}_{agent_id}')
                    # update to last state
                    last_obs[agent_id] = cur_ob
                    last_phases[agent_id] = cur_phase
                    dones[agent_id] = done

                    if total_decision_num > trainer.learning_start and\
                        total_decision_num % trainer.update_model_rate == trainer.update_model_rate - 1:

                        episode_loss[agent_id].append(trainer.agents[agent_id].train())
                    # TODO: combine learn and update rate into agent: this is more reasonable
                    if total_decision_num > trainer.learning_start and \
                        total_decision_num % trainer.update_target_rate == trainer.update_target_rate - 1:
                        trainer.agents[agent_id].update_target_network()

                    if flag:
                        # store all actions in one step
                        f_actions = deepcopy(actions)
                        f_actions_prob = deepcopy(actions_prob)
                        trainer.metric.update(np.array([acc_rewards[ag]/trainer.action_interval for ag in trainer.agents]))
                        # TODO: trainer.metric.update(rewards)
                        acc_rewards = {agent_id: 0 for agent_id in trainer.agents}
                        total_decision_num += 1
                        i += 1
                        flag = False
                        break
                if all(dones.values()) == True:
                    break
            else:
                for agent_id in trainer.env.agent_iter():
                    if trainer.env._agent_selector.is_last():
                        flag = True
                    _, _, reward, done, _ = trainer.env.last()
                    acc_rewards[agent_id] += reward
                    dones[agent_id] = done
                    trainer.env.step(f_actions[agent_id])
                    if flag:
                        i += 1
                        flag = False
                        break
                if all(dones.values()) == True:
                    break

        if all([loss_v for loss_v in episode_loss.values()]):
            mean_loss = np.mean(np.array([loss_v for loss_v in episode_loss.values()]))
        else:
            mean_loss = 0
    
        trainer.writeLog("TRAIN", e, trainer.metric.real_average_travel_time(),\
            mean_loss, trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), trainer.metric.throughput())
        trainer.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, trainer.steps,\
            mean_loss, trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), int(trainer.metric.throughput())))
        if e % trainer.save_rate == 0:
            [ag.save_model(e=e) for ag in trainer.agents.values()]
        trainer.logger.info("episode:{}/{}, real avg travel time:{}".format(e, trainer.episodes, trainer.metric.real_average_travel_time()))
        for j in range(len(trainer.world.intersections)):
            trainer.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, trainer.metric.lane_rewards()[j],\
                 trainer.metric.lane_queue()[j]))
        if trainer.test_when_train:
            trainer.train_test(e)
    # trainer.dataset.flush([ag.replay_buffer for ag in trainer.agents])
    [ag.save_model(e=trainer.episodes) for ag in trainer.agents.values()]

def tsc_ma_test_helper(trainer):
    i = 0
    flag = False
    trainer.env.reset()
    dones = {agent_id: None for agent_id in trainer.agents}
    acc_rewards = {agent_id: 0 for agent_id in trainer.agents}

    while i < trainer.steps:
        if i % trainer.action_interval == 0:
            # Clear last round actions
            actions = {agent_id: None for agent_id in trainer.agents}

            for agent_id in trainer.env.agent_iter():
                if trainer.env._agent_selector.is_last():
                    flag = True
                cur_ob, cur_phase, reward, done, _ = trainer.env.last()
                acc_rewards[agent_id] += reward
                action = trainer.agents[agent_id].get_action(cur_ob, cur_phase, test=False)
                actions[agent_id] = action
                trainer.env.step(actions[agent_id])
                dones[agent_id] = done
                
                if flag:
                    f_actions = deepcopy(actions)
                    trainer.metric.update(np.array([acc_rewards[ag]/trainer.action_interval for ag in trainer.agents]))
                    acc_rewards = {agent_id: 0 for agent_id in trainer.agents}
                    i += 1
                    flag = False
                    break
            if all(dones.values()) == True:
                break
        else:
            for agent_id in trainer.env.agent_iter():
                if trainer.env._agent_selector.is_last():
                    flag = True
                cur_ob, cur_phase, reward, done, _ = trainer.env.last()
                acc_rewards[agent_id] += reward
                dones[agent_id] = done
                trainer.env.step(actions[agent_id])
                if flag:
                    i += 1
                    flag = False
                    break
            if all(dones.values()) == True:
                break

def tsc_ma_train_test(trainer, e):
    '''
    train_test
    Evaluate model performance after each episode training process.
    :param e: number of episode
    :return trainer.metric.real_average_travel_time: travel time of vehicles
    '''
    tsc_ma_test_helper(trainer)
    trainer.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
        e, trainer.episodes, trainer.metric.real_average_travel_time(), trainer.metric.rewards(),\
        trainer.metric.queue(), trainer.metric.delay(), int(trainer.metric.throughput())))
    trainer.writeLog("TEST", e, trainer.metric.real_average_travel_time(),\
        100, trainer.metric.rewards(),trainer.metric.queue(),trainer.metric.delay(), trainer.metric.throughput())
    return trainer.metric.real_average_travel_time()

def tsc_ma_test(trainer, drop_load=True):
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

    # TODO: addin model loader later
    # if not drop_load:
    #     [ag.load_model(trainer.episodes) for ag in trainer.agents]

    tsc_ma_test_helper(trainer)
    trainer.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (trainer.metric.real_average_travel_time(), \
        trainer.metric.rewards(), trainer.metric.queue(), trainer.metric.delay(), trainer.metric.throughput()))
    return trainer.metric