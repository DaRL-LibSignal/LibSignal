import gym
from environment_colight import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.colight_agent import CoLightAgent
from metric import TravelTimeMetric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
from utils import *
import pickle


# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str,help='path of config file')  #road net
parser.add_argument('--thread', type=int, default=1,help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="2",help='gpu to be used')  # choose gpu card
parser.add_argument('-lr','--learning_rate', type=float, default=1e-3, help="learning rate")
parser.add_argument('-bs','--batch_size', type=int, default=64, help="batch size")
parser.add_argument('-ls','--learning_start', type=int, default=1000, help="learning start")
parser.add_argument('-rs','--replay_buffer_size', type=int, default=5000, help="size of replay buffer")
parser.add_argument('-uf','--update_target_model_freq', type=int, default=10, help="the frequency to update target q model")
parser.add_argument('-pr','--prefix', type=str, default="yzy1", help="the prefix of model and file")
parser.add_argument('-gc','--grad_clip', type=float, default=5.0, help="clip gradients")
parser.add_argument('-ep','--epsilon', type=float, default=0.8, help="exploration rate")
parser.add_argument('-ed','--epsilon_decay', type=float, default=0.9995, help="decay rate of exploration rate")
parser.add_argument('-me','--min_epsilon', type=float, default=0.01, help="the minimum epsilon when decaying")
parser.add_argument('--steps', type=int, default=3600,help='number of steps')  #per episodes
parser.add_argument('--test_steps', type=int,default=3600,help='number of steps for step')
parser.add_argument('--action_interval',type=int,default=10,help='how often agent make decisions')
parser.add_argument('--episodes',type=int,default=200,help='training episodes')
#parser.add_argument('--test_episodes',type=int,default=10,help='testing episodes')
parser.add_argument('--load_model_dir',type=str,default=None,help='load this model to test')
parser.add_argument('--graph_info_dir',type=str,default="syn33",help='load infos about graph(i.e. mapping, adjacent)')
parser.add_argument('--train_model', action="store_false", default=True)
parser.add_argument('--test_model', action="store_true", default=False)
parser.add_argument('--save_model', action="store_false", default=True)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate",type=int,default=10,help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir',type=str,default="model/colight_debug_1x1",help='directory in which model should be saved')
#parser.add_argument('--load_dir',type=str,default="model/colight",help='directory in which model should be loaded')
parser.add_argument('--log_dir',type=str,default="log/colight_debug_1x1",help='directory in which logs should be saved')
parser.add_argument('--vehicle_max',type=int,default=1,help='used to normalize node observayion')
parser.add_argument('--mask_type',type=int,default=0,help='used to specify the type of softmax')
parser.add_argument('--get_attention', action="store_true", default=False)
parser.add_argument('--test_when_train', action="store_false", default=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
file_prefix = args.prefix + "_" + "Colight_" + str(args.graph_info_dir) + "_" + str(args.learning_rate) + "_" + str(
    args.epsilon) + "_" + str(args.epsilon_decay) + "_" + str(args.batch_size) + "_" + str(
    args.learning_start) + "_" + str(args.replay_buffer_size) + "_" + datetime.now().strftime('%Y%m%d-%H%M%S')
fh = logging.FileHandler(os.path.join(args.log_dir, file_prefix + ".log"))
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

# create world
world0 = World(args.config_file, thread_num=args.thread)
graph_info_file_dir = args.graph_info_dir + ".pkl"
graph_info_file = open(graph_info_file_dir, "rb")
res = pickle.load(graph_info_file)
net_node_dict_id2inter = res[0]
net_node_dict_inter2id = res[1]
net_edge_dict_id2edge = res[2]
net_edge_dict_edge2id = res[3]
node_degree_node = res[4]
node_degree_edge = res[5]
node_adjacent_node_matrix = res[6]
node_adjacent_edge_matrix = res[7]
edge_adjacent_node_matrix = res[8]
# net_node_dict_id2inter, net_node_dict_inter2id, net_edge_dict_id2edge, net_edge_dict_edge2id, \
#     node_degree_node,node_degree_edge, node_adjacent_node_matrix, node_adjacent_edge_matrix, \
#     edge_adjacent_node_matrix = pickle.load(graph_info_file)
graph_info_file.close()
# TODO:update the below dict (already done)
dic_traffic_env_conf = {
    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": len(net_node_dict_id2inter),  # used
    "NUM_ROADS": len(net_edge_dict_id2edge),  # used
    "MIN_ACTION_TIME": 10,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 8,  # used
    "NUM_LANES": 1,  # used
    "ACTION_DIM": 2,
    "MEASURE_TIME": 10,
    "IF_GUI": True,
    "DEBUG": False,
    "INTERVAL": 1,
    "THREADNUM": 8,
    "SAVEREPLAY": True,
    "RLTRAFFICLIGHT": True,
    "DIC_FEATURE_DIM": dict(  # used
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),
        D_COMING_VEHICLE=(4,),
        D_LEAVING_VEHICLE=(4,),
        D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
        D_CUR_PHASE=(1,),  # used
        D_NEXT_PHASE=(1,),
        D_TIME_THIS_PHASE=(1,),
        D_TERMINAL=(1,),
        D_LANE_SUM_WAITING_TIME=(4,),
        D_VEHICLE_POSITION_IMG=(4, 60,),
        D_VEHICLE_SPEED_IMG=(4, 60,),
        D_VEHICLE_WAITING_TIME_IMG=(4, 60,),
        D_PRESSURE=(1,),
        D_ADJACENCY_MATRIX=(2,)),
    # used
    "LIST_STATE_FEATURE": [
        "cur_phase",
        # "time_this_phase",
        # "vehicle_position_img",
        # "vehicle_speed_img",
        # "vehicle_acceleration_img",
        # "vehicle_waiting_time_img",
        "lane_num_vehicle",
        # "lane_num_vehicle_been_stopped_thres01",
        # "lane_num_vehicle_been_stopped_thres1",
        # "lane_queue_length",
        # "lane_num_vehicle_left",
        # "lane_sum_duration_vehicle_left",
        # "lane_sum_waiting_time",
        # "terminal",

        # "coming_vehicle",
        # "leaving_vehicle",
        # "pressure",

        # "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
        "sum_duration_vehicle_left": 0,
        "sum_num_vehicle_been_stopped_thres01": 0,
        "sum_num_vehicle_been_stopped_thres1": -0.25,
        "pressure": 0,
    },
    "LANE_NUM": {
        "LEFT": 1,
        "RIGHT": 1,
        "STRAIGHT": 1
    },
    "PHASE": [
        'WSES',
        'NSSS',
        'WLEL',
        'NLSL',
        'WSWL',
        'ESEL',
        'NSNL',
        'SSSL',
    ],
}

dic_graph_setting = {
    "NEIGHBOR_NUM": 4,  # standard number of adjacent nodes of each node
    "NEIGHBOR_EDGE_NUM": 4,  # # standard number of adjacent edges of each node
    "N_LAYERS": 1,  # layers of MPNN
    "INPUT_DIM": [128, 128],
    # input dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
    "OUTPUT_DIM": [128, 128],
    # output dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
    "NODE_EMB_DIM": [128, 128],  # the firsr two layer of dense to embedding the input
    "NUM_HEADS": [5, 5],
    "NODE_LAYER_DIMS_EACH_HEAD": [16, 16],  # [input_dim,output_dim]
    "OUTPUT_LAYERS": [],  #
    "NEIGHBOR_ID": node_adjacent_node_matrix,  # adjacent node id of each node
    "ID2INTER_MAPPING": net_node_dict_id2inter,  # id ---> intersection mapping
    "INTER2ID_MAPPING": net_node_dict_inter2id,  # intersection ----->id mapping
    "NODE_DEGREE_NODE": node_degree_node,  # number of adjacent nodes of node
}


def build(path):
    world = World(path, thread_num=args.thread)
    # create observation generator, which is used to construct sample
    observation_generators = []
    for node_dict in world.intersections:
        node_id = node_dict.id
        node_id_int = net_node_dict_inter2id[node_id]
        tmp_generator = LaneVehicleGenerator(world,
                                             node_dict, ["lane_count"],
                                             in_only=True,
                                             average='road')
        observation_generators.append((node_id_int, tmp_generator))
        # if len(observation_generators) == 5:
        #     break
    sorted(observation_generators,
           key=lambda x: x[0])  # sorted the ob_generator based on its corresponding id_int, increasingly

    # create agent
    action_space = gym.spaces.Discrete(len(world.intersections[0].phases))
    colightAgent = CoLightAgent(
        action_space, observation_generators,
        LaneVehicleGenerator(world, world.intersections[0], ["lane_waiting_count"], in_only=True, average="all",
                             negative=True), world, dic_traffic_env_conf, dic_graph_setting, args)
    if args.load_model:
        colightAgent.load_model(args.load_dir)
    print(colightAgent.ob_length)
    print(colightAgent.action_space)
    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, colightAgent, metric)
    return world, colightAgent, env

# train colight_agent
class TrafficLightDQN:
    def __init__(self, world, agent, env, args, logging_tool, fprefix):
        self.world, self.agent, self.env = build(args.config_file)
        self.logging_tool = logging_tool
        self.yellow_time = self.world.intersections[0].yellow_phase_time
        self.args = args
        self.fprefix = fprefix
        # self.log_file = os.path.join(self.args.log_dir,self.args.prefix+ datetime.now().strftime('%Y%m%d-%H%M%S') + ".yzy.log")
        self.log_file=os.path.join(args.log_dir,self.fprefix+".yzy.log")
        # self.log_file = file_prefix + ".yzy.log"
        log_handle = open(self.log_file, 'w')
        log_handle.close()
        self.replay_file_dir = "data/replay_dir/"+self.args.config_file
        if not os.path.exists(self.replay_file_dir):
            os.makedirs(self.replay_file_dir)
        self.replay_file_dir = "replay_dir/"+self.args.config_file

    def train(self):
        total_decision_num = 0
        for e in range(self.args.episodes):
            last_obs = self.env.reset()  # observation dimension?
            if e % self.args.save_rate == self.args.save_rate - 1:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(self.replay_file_dir+"/%s_replay_%s.txt" % (self.fprefix,e))
            else:
                self.env.eng.set_save_replay(False)
            episodes_rewards = [0 for i in range(len(self.world.intersections))]
            episodes_decision_num = 0
            episode_loss = []
            i = 0
            while i < self.args.steps:
                if i % self.args.action_interval == 0:
                    # print(i)
                    actions = []
                    last_phase = [] # ordered by the int id of intersections
                    for j in range(len(self.world.intersections)):
                        node_id_str = self.agent.graph_setting["ID2INTER_MAPPING"][j]
                        node_dict = self.world.id2intersection[node_id_str]
                        last_phase.append(node_dict.current_phase)
                        # last_phase.append([self.world.intersections[j].current_phase])
                    if (total_decision_num > self.agent.learning_start):
                        actions = self.agent.get_action(last_phase, last_obs)  
                        # the retured dimension is [batch, agents],
                        # the batch is 1 when we get action, so we just get the first actions
                        actions = actions[0]
                    else:
                        actions = self.agent.sample(s_size=self.agent.num_agents)
                    reward_list = []  # [intervals,agents,reward]
                    # actions[len(actions) - 1] = 0
                    # print(actions)
                    for _ in range(self.args.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions)
                        i += 1
                        reward_list.append(rewards)
                    rewards = np.mean(reward_list, axis=0) # [agents,reward]
                    for j in range(len(self.world.intersections)):
                        episodes_rewards[j] += rewards[j]
                    cur_phase = []
                    for j in range(len(self.world.intersections)):
                        node_id_str = self.agent.graph_setting["ID2INTER_MAPPING"][j]
                        node_dict = self.world.id2intersection[node_id_str]
                        cur_phase.append(node_dict.current_phase)
                    self.agent.remember(last_obs, last_phase, actions, rewards, obs, cur_phase)
                    episodes_decision_num += 1
                    total_decision_num += 1
                    last_obs = obs

                if total_decision_num > self.agent.learning_start and total_decision_num % self.agent.update_model_freq == self.agent.update_model_freq - 1:
                    cur_loss_q = self.agent.replay()
                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.agent.learning_start and total_decision_num % self.agent.update_target_model_freq == self.agent.update_target_model_freq - 1:
                    self.agent.update_target_network()
                if all(dones):
                    break
            if len(episode_loss)>0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            cur_travel_time = self.env.eng.get_average_travel_time()
            mean_reward = np.sum(episodes_rewards)/episodes_decision_num
            self.writeLog("TRAIN",e,cur_travel_time,mean_loss,mean_reward)
            self.logging_tool.info("step:{}/{}, q_loss:{}, rewards:{}".format(i, self.args.steps, mean_loss,mean_reward))
            if e % self.args.save_rate == self.args.save_rate - 1:
                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir)
                self.agent.save_model(e, self.fprefix, self.args.save_dir)
            self.logging_tool.info("episode:{}/{}, average travel time:{}".format(e, self.args.episodes, cur_travel_time))
            for j in range(len(self.world.intersections)):
                self.logging_tool.debug("intersection:{}, mean_episode_reward:{}".format(j, episodes_rewards[j] / episodes_decision_num))
            if self.args.test_when_train:
                self.train_test(e)
        self.agent.save_model(self.args.episodes, self.fprefix, self.args.save_dir)

    def train_test(self,e):
        obs = self.env.reset()
        ep_rwds = [0 for i in range(len(self.world.intersections))]
        eps_nums = 0
        for i in range(self.args.test_steps):
            if i % args.action_interval == 0:
                last_phase = []
                for j in range(len(self.world.intersections)):
                    node_id_str = self.agent.graph_setting["ID2INTER_MAPPING"][j]
                    node_dict = self.world.id2intersection[node_id_str]
                    last_phase.append(node_dict.current_phase)
                actions = self.agent.get_action(last_phase, obs,test_phase=True)
                actions = actions[0]
                rewards_list = []
                # actions[len(actions) - 1] = 0
                for _ in range(self.args.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                for j in range(len(self.world.intersections)):
                        ep_rwds[j] += rewards[j]
                eps_nums+=1
            if all(dones):
                break
        mean_rwd = np.sum(ep_rwds)/eps_nums 
        trv_time = self.env.eng.get_average_travel_time()
        #self.logging_tool.info("Final Travel Time is %.4f, and mean rewards %.4f" % (trv_time,mean_rwd))
        self.logging_tool.info("Test step:{}/{}, travel time :{}, rewards:{}".format(e, self.args.episodes, trv_time, mean_rwd))
        self.writeLog("TEST",e,trv_time,100,mean_rwd)
        return trv_time

    def test(self,drop_load=False):
        if not drop_load:
            model_name=self.args.save_dir
            if model_name is not None:
                self.agent.load_model(model_name)
            else:
                raise ValueError("model name should not be none")
        attention_mat_list = []
        obs = self.env.reset()
        ep_rwds = [0 for i in range(len(self.world.intersections))]
        eps_nums = 0
        for i in range(self.args.test_steps):
            if i % args.action_interval == 0:
                last_phase = []
                for j in range(len(self.world.intersections)):
                    node_id_str = self.agent.graph_setting["ID2INTER_MAPPING"][j]
                    node_dict = self.world.id2intersection[node_id_str]
                    last_phase.append(node_dict.current_phase)
                if self.args.get_attention:
                    actions, att_step = self.agent.get_action(last_phase, obs,test_phase=True)
                    attention_mat_list.append(att_step[0])
                else:
                    actions = self.agent.get_action(last_phase, obs,test_phase=True)
                actions = actions[0]
                rewards_list = []
                # actions[len(actions) - 1] = 0
                for _ in range(self.args.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                for j in range(len(self.world.intersections)):
                        ep_rwds[j] += rewards[j]
                eps_nums+=1
                #ep_rwds.append(rewards)

            # print(env.eng.get_average_travel_time())
            if all(dones):
                break
        mean_rwd = np.sum(ep_rwds)/eps_nums 
        trv_time = self.env.eng.get_average_travel_time()
        self.logging_tool.info("Final Travel Time is %.4f, and mean rewards %.4f" % (trv_time,mean_rwd))
        if self.args.get_attention:
            tmpstr = self.args.load_model_dir
            tmpstr=tmpstr.split('/')[-1]
            att_file = "data/analysis/colight/"+tmpstr+"_att_ana.pkl"
            pickle.dump(attention_mat_list,open(att_file,"wb"))
            print("dump the attention matrix to ",att_file)
        return trv_time

    def writeLog(self, mode, step, travel_time, loss, cur_rwd):
        """                                                                           
        :param mode: "TRAIN" OR "TEST"                                                
        :param step: int                                                              
        """
        res = "CoLight" + '\t' + mode + '\t' + str(step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss +"\t"+ "%.2f" % cur_rwd
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()


if __name__ == '__main__':
    world, colightAgent, env = build(args.config_file)
    player = TrafficLightDQN(world, colightAgent, env , args, logger,file_prefix)
    if args.train_model:
        # print("begin to train model")
        player.train()
        player.test(False)
    if args.test_model:
        print(args.load_model_dir)
        if (not args.train_model) and (args.load_model_dir is None):
            raise ValueError("invalid parameters, load_model_dir should not be None when the agent is not trained")
        print("begin to test model")
        player.test()
# simulate
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# train(args, env)
# test()
# meta_test('/mnt/d/Cityflow/examples/config.json')