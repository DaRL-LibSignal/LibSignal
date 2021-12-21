from . import RLAgent
import random
import numpy as np
import tensorflow as tf
from collections import deque
import os
import pickle
import time
import keras
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model, model_from_json, load_model
from keras.layers.core import Activation
from keras.utils import np_utils,to_categorical
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, TensorBoard


class RepeatVector3D(Layer):
    """
    expand axis=1, then tile times on axis=1
    """
    def __init__(self,times,**kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.times, input_shape[1],input_shape[2])

    def call(self, inputs):
        #[batch,agent,dim]->[batch,1,agent,dim]
        #[batch,1,agent,dim]->[batch,agent,agent,dim]
        return K.tile(K.expand_dims(inputs,1),[1,self.times,1,1])


    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CoLightAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, world, traffic_env_conf, graph_setting, args):
        super().__init__(action_space, ob_generator[0][1], reward_generator)

        self.ob_generators = ob_generator # a list of ob_generators for each intersection ordered by its int_id
        self.ob_length = ob_generator[0][1].ob_length # the observation length for each intersection
        
        self.graph_setting = graph_setting
        self.neighbor_id = self.graph_setting["NEIGHBOR_ID"] # neighbor node of node
        self.degree_num = self.graph_setting["NODE_DEGREE_NODE"] # degree of each intersection
        self.direction_dic = {"0":[1,0,0,0],"1":[0,1,0,0],"2":[0,0,1,0],"3":[0,0,0,1]} # TODO: should refine it

        self.dic_traffic_env_conf = traffic_env_conf
        self.num_agents=self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_edges = self.dic_traffic_env_conf["NUM_ROADS"]

        self.vehicle_max = args.vehicle_max
        self.mask_type = args.mask_type
        self.get_attention = args.get_attention

        #DQN setting
        self.batch_size = args.batch_size #32
        self.learning_rate = args.learning_rate #0.0001
        self.replay_buffer = deque(maxlen=args.replay_buffer_size)
        self.learning_start = args.learning_start #2000
        self.update_model_freq = 1
        self.update_target_model_freq = args.update_target_model_freq #20
        self.gamma = 0.95  # discount rate
        self.epsilon = args.epsilon  # exploration rate
        self.epsilon_min = args.min_epsilon #default 0.01
        self.epsilon_decay = args.epsilon_decay # 0.995
        self.grad_clip = args.grad_clip

        self.world = world
        self.world.subscribe("pressure")
        self.world.subscribe("lane_count")
        self.world.subscribe("lane_waiting_count")
        
        self._placeholder_init()
        self._build_eval_model()
        self._build_target_model()
        self.eval_params = tf.get_collection("eval_q_model")
        self.target_params = tf.get_collection("target_q_model")
        self.replace_target_op = [tf.assign(t,e) for t,e in zip(self.target_params,self.eval_params)]
        t_config = tf.ConfigProto()
        t_config.gpu_options.allow_growth = True
        self.algo_saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=2)
        self.sess = tf.Session(config=t_config)
        self.sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./log/', tf.get_default_graph())
        self.sess.run(self.replace_target_op)   

    def _placeholder_init(self):
        self.node_state_length = 0
        # for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
        #     if  "adjacency" in feature_name:
        #         self.node_state_length += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0]
        #     else:
        #         self.node_state_length += self.ob_length
        self.input_node_state=tf.placeholder(dtype=tf.float32, shape=[None,self.num_agents,self.ob_length],name='input_state') # concat [lane_num_vehicle]
        self.input_node_phase=tf.placeholder(dtype=tf.int32, shape=[None,self.num_agents],name='input_phase')
        self.input_neighbor_id = tf.placeholder(dtype=tf.int32, shape=[self.num_agents,self.graph_setting["NEIGHBOR_NUM"]],name="neighbor_id") #neighbor node of node
        self.input_node_degree_mask = tf.placeholder(dtype=tf.int32,shape=[self.num_agents], name="node_degree_mask") #not all nodes has degree 4
        self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None,self.num_agents], name="input_actions") #batch ,agents
        self.input_target_value = tf.placeholder(dtype=tf.float32, shape=[None,self.num_agents,],name="input_target_value")
        one_hot_phase = tf.one_hot(self.input_node_phase, len(self.world.intersections[0].phases)) # batch, agents, num_phases
        self.concat_input = tf.concat([self.input_node_state,one_hot_phase],axis=-1,name="concate_input_obs")
        #actions one-hot
        self.input_actions_one_hot = tf.one_hot(self.input_actions, self.action_space.n)
        # the adjacent node of node
        #self.neighbor_node_one_hot = tf.one_hot(self.input_neighbor_id,self.num_agents) # agent, neighbor, agents  
        neighbor_node_tmp = tf.range(0,self.num_agents)
        neighbor_node_tmp = tf.expand_dims(neighbor_node_tmp,axis=-1)
        expanded_neighbor_id = tf.concat([neighbor_node_tmp,self.input_neighbor_id],axis=-1) #[agents ,num_neighbor+1] include node itself
        neighbor_node_one_hot = tf.one_hot(expanded_neighbor_id,self.num_agents)
        neighbor_node_one_hot = tf.expand_dims(neighbor_node_one_hot,axis=0)
        self.neighbor_node_one_hot = tf.tile(neighbor_node_one_hot,[tf.shape(self.input_node_state)[0],1,1,1])
        # process the mask
        degree_mask = self.input_node_degree_mask + 1 
        degree_mask = tf.sequence_mask(degree_mask,self.graph_setting["NEIGHBOR_NUM"]+1) # agetns, neighbor_num+1
        # degree_mask = tf.expand_dims(degree_mask,axis=-1) 
        self.degree_mask = tf.cast(degree_mask,tf.float32,name="processed_degree") #agents, neighbor_num

    # def _state_embedding(self):
    #     with tf.variable_scope("First_Embedding",reuse=tf.AUTO_REUSE):
    #         self.phase_embedding_matrix = tf.get_variable(name="phase_emb_matrix",shape=[len(self.world.intersections[0].phases),self.graph_setting["PHASE_EMB_DIM"]])
    
    def _build_model(self):
        """
        layer definition
        """
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,neighbors,agents]
        # In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        # In.append(Input(shape=(self.num_agents,self.num_neighbors,self.num_agents),name="adjacency_matrix")) 

        feature=self.MLP(self.input_node_state,self.graph_setting["NODE_EMB_DIM"]) #feature:[batch,agents,feature_dim]
        
        att_record_all_layers=list()
        for i in range(self.graph_setting["N_LAYERS"]):
            if i==0:
                h,att_record=self._MultiHeadsAttModel(
                    feature,
                    self.neighbor_node_one_hot,
                    l=self.graph_setting["NEIGHBOR_NUM"],
                    d=self.graph_setting["INPUT_DIM"][i],
                    dv=self.graph_setting["NODE_LAYER_DIMS_EACH_HEAD"][i],
                    dout=self.graph_setting["OUTPUT_DIM"][i],
                    nv=self.graph_setting["NUM_HEADS"][i],
                    suffix=i
                    )
            else:
                h,att_record=self._MultiHeadsAttModel(
                    h,
                    self.neighbor_node_one_hot,
                    l=self.graph_setting["NEIGHBOR_NUM"],
                    d=self.graph_setting["INPUT_DIM"][i],
                    dv=self.graph_setting["NODE_LAYER_DIMS_EACH_HEAD"][i],
                    dout=self.graph_setting["OUTPUT_DIM"][i],
                    nv=self.graph_setting["NUM_HEADS"][i],
                    suffix=i
                    )
            att_record_all_layers.append(att_record)
        # if self.graph_setting["N_LAYERS"]>1:
        #     att_record_all_layers=Concatenate(axis=1)(att_record_all_layers)
        # else:
        #     att_record_all_layers=att_record_all_layers[0]

        # att_record_all_layers=Reshape((self.graph_setting["N_LAYERS"],self.num_agents,self.graph_setting["NUM_HEADS"][self.graph_setting["N_LAYERS"]-1],self.graph_setting["NEIGHBOR_NUM"]+1))(att_record_all_layers)
        for layer_index,layer_size in enumerate(self.graph_setting["OUTPUT_LAYERS"]):
                h=Dense(layer_size,activation='relu',kernel_initializer='random_normal',name='Dense_q_%d'%layer_index)(h)
        # out = Dense(self.action_space.n,kernel_initializer='random_normal',name='action_layer')(h)
        # out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        # model=Model(inputs=In,outputs=[out,att_record_all_layers])
        return h,att_record_all_layers

    def _build_target_model(self):
        with tf.variable_scope("target_q_model",reuse=tf.AUTO_REUSE):
            value_output, _ = self._build_model()
            self.target_q_value=Dense(self.action_space.n,kernel_initializer='random_normal',name='target_q_value')(value_output)
            #self.target_q_value = tf.layers.dense(value_output,units=self.action_space.n, name="target_q_value") #batch agents value


    def _build_eval_model(self):
        with tf.variable_scope("eval_q_model",reuse=tf.AUTO_REUSE):
            value_output, att_record_eval = self._build_model()
            self.value = Dense(self.action_space.n,kernel_initializer='random_normal',name='q_value')(value_output)
            self.attention_record = att_record_eval[-1] #[batch,agents,nv,neighbor], used in visualize when testing, generally its only 1 gat layer
            # self.value = tf.layers.dense(value_output,units=self.action_space.n, name="q_value") #batch agents action_dim
            q_value = tf.reduce_sum(self.value*self.input_actions_one_hot,axis=-1) # batch agents
            self.q_loss= tf.reduce_mean(tf.square(q_value-self.input_target_value),name="q_loss")
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ =tf.clip_by_global_norm(tf.gradients(self.q_loss,tvars),self.grad_clip)
            self.train_op=self.optimizer.apply_gradients(zip(grads,tvars),name="train_op")
            #self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.q_loss)

    def _MultiHeadsAttModel(self,In_agent,In_neighbor,l=5, d=128, dv=16, dout=128, nv = 8,suffix=-1):
        """
        input:
            In_agent [bacth,agent,128]
            In_neighbor [agent, neighbor_num]
            l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
            d: dimension of agent's embedding
            dv: dimension of each head
            dout: dimension of output
            nv: number of head (multi-head attention)
        output:
            -hidden state: [batch,agent,32]
            -attention: [batch,agent,neighbor]
        """

        """agent repr"""
        print("In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv", In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv)
        #[batch,agent,dim]->[batch,agent,1,dim]
        agent_repr=Reshape((self.num_agents,1,d))(In_agent)

        """neighbor repr"""
        #[batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr=RepeatVector3D(self.num_agents)(In_agent)
        print("neighbor_repr.shape", neighbor_repr.shape)
        #[batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr=Lambda(lambda x:K.batch_dot(x[0],x[1]))([In_neighbor,neighbor_repr])
        print("neighbor_repr.shape", neighbor_repr.shape)
        
        """attention computation"""
        #multi-head
        #[batch,agent,1,dim]->[batch,agent,1,dv*nv]
        agent_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='agent_repr_%d'%suffix)(agent_repr)
        #[batch,agent,1,dv,nv]->[batch,agent,nv,1,dv]
        agent_repr_head=Reshape((self.num_agents,1,dv,nv))(agent_repr_head)
        agent_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(agent_repr_head)

        neighbor_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_repr_%d'%suffix)(neighbor_repr)
        #[batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        print("DEBUG",neighbor_repr_head.shape)
        print("self.num_agents,self.num_neighbors,dv,nv", self.num_agents,self.graph_setting["NEIGHBOR_NUM"],dv,nv)
        neighbor_repr_head=Reshape((self.num_agents,self.graph_setting["NEIGHBOR_NUM"]+1,dv,nv))(neighbor_repr_head)
        neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_repr_head)
        
        #should mask
        tmp_mask = tf.expand_dims(self.degree_mask,axis=1) # [agents,neighbor] --->  [agents,1,neighbor]
        tmp_mask = tf.expand_dims(tmp_mask,axis=-2) # [agents,1,neighbor] --->  [agents,1,1,neighbor]
        tmp_mask = tf.tile(tmp_mask,[1,nv,1,1]) # [agents,1,1,neighbor] --->  [agents,nv,1,neighbor]

        if self.mask_type==1:
            # [batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
            # neighbor_repr_head=Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[4,4]))([agent_repr_head,neighbor_repr_head])
            # tmp_neigbor_repr_head = neighbor_repr_head - tf.stop_gradient(tf.expand_dims(tf.reduce_max(neighbor_repr_head,axis=-1),axis=-1))
            # tmp_neigbor_repr_head = neighbor_repr_head - tf.expand_dims(tf.reduce_max(neighbor_repr_head,axis=-1),axis=-1)
            # neighbor_repr_head = tf.exp(tmp_neigbor_repr_head)
            # neighbor_repr_head=neighbor_repr_head * tmp_mask
            # tmp_sum = tf.reduce_sum(neighbor_repr_head,axis=-1) # [batch,agent,nv,1]
            # tmp_sum = tf.expand_dims(tmp_sum,axis=-1)# [batch,agent,nv,1,1]
            # att = neighbor_repr_head / tmp_sum

            #[batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
            att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head]) #[batch,agents,nv,1,neighbor]
            att = att * tmp_mask #[batch,agents,nv,1,neighbor]
            tmp_att = att
            att = att / tf.expand_dims(tf.reduce_sum(tmp_att,axis=-1),axis=-1)
            print("att.shape:",att.shape)
        else:
            #[batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
            att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head]) #[batch,agents,nv,1,neighbor]
            att = att * tmp_mask #[batch,agents,nv,1,neighbor]
            print("att.shape:",att.shape)
        att_record=Reshape((self.num_agents,nv,self.graph_setting["NEIGHBOR_NUM"]+1))(att) #[batch,agent,nv,1,neighbor]->[batch,agent,nv,neighbor]
        


        #self embedding again
        neighbor_hidden_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_hidden_repr_%d'%suffix)(neighbor_repr)
        neighbor_hidden_repr_head=Reshape((self.num_agents,self.graph_setting["NEIGHBOR_NUM"]+1,dv,nv))(neighbor_hidden_repr_head) #[batch,agents,neighbor,dv,nv]
        neighbor_hidden_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_hidden_repr_head) #[batch,agents,nv,neighbor,dv]
        out=Lambda(lambda x:K.mean(K.batch_dot(x[0],x[1]),axis=2))([att,neighbor_hidden_repr_head]) # [batch,agents,nv,1,neighbor]*[batch,agents,nv,neighbor,dv]--->[batch,agents,nv,1,dv]--->[batch,agents,1,dv]
        print("out-shape:",out.shape)
        out=Reshape((self.num_agents,dv))(out) #[batch, agents,dv] 
        out = Dense(dout, activation = "relu",kernel_initializer='random_normal',name='MLP_after_relation_%d'%suffix)(out)
        return out,att_record
    
    def MLP(self,In_0,layers=[128,128]):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,feature_dim]
        -outpout: [batch,#agents,128]
        """
        # In_0 = Input(shape=[self.num_agents,self.len_feature])
        for layer_index,layer_size in enumerate(layers):
            if layer_index==0:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(In_0)
            else:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(h)

        return h

    def get_action(self, phase, ob, test_phase=False):
        """
        phase : [agents]
        obs : [agents, ob_length]
        edge_obs : [agents, edge_ob_length]
        return: [batch,agents]
        we should expand the input first to adapt the [batch,] standard
        """
        if not test_phase:
            #print("train_phase")
            if np.random.rand() <= self.epsilon:
                return self.sample([1,self.num_agents])
        # ob = self._reshape_ob(ob)
        # act_values = self.model.predict([phase, ob])
        e_ob = ob[np.newaxis,:]
        e_phase = np.array(phase)
        e_phase = e_phase[np.newaxis,:]
        #observations = np.concatenate([phase,ob],axis=-1)
        #observations = observations[np.newaxis,:]
        my_feed_dict = {self.input_node_state:e_ob, self.input_node_phase:e_phase, self.input_neighbor_id:self.neighbor_id,self.input_node_degree_mask: self.degree_num}
        if self.get_attention:
            act_values, att_mat = self.sess.run([self.value,self.attention_record],feed_dict=my_feed_dict)
            return np.argmax(act_values, axis=-1), att_mat #[batch, agents],[batch,agent,nv,neighbor]
        else:
            act_values = self.sess.run(self.value,feed_dict=my_feed_dict)
            return np.argmax(act_values, axis=-1) #batch, agents
        
    def sample(self,s_size):
        return np.random.randint(0,self.action_space.n,s_size)
        #return self.action_space.sample()

    def get_reward(self, reward_type="vehicleNums"):
        #reward_type should in ["pressure", "vehicleNums"]
        #The colight use queue length on the approaching lane l at time t (minus)
        #lane_nums = self.world.get_info("")
        # should care about the order of world.intersection
        if reward_type=="pressure":
            pressures = self.world.get_info("pressure") # we should change the dict pressure to list
            pressure_reward = [] # order by the int_id of agents
            for i in range(self.num_agents):
                node_id = self.graph_setting["ID2INTER_MAPPING"][i]
                pressure_reward.append(-pressures[node_id])
            pressure_reward = np.array(pressure_reward)
            return pressure_reward
        elif reward_type=="vehicleNums":
            vehicle_reward = []
            vehicle_nums = self.world.get_info("lane_waiting_count")
            for i in range(self.num_agents):
                node_id = self.graph_setting["ID2INTER_MAPPING"][i]
                node_dict = self.world.id2intersection[node_id]
                nvehicles=0
                tmp_vehicle = []
                in_lanes=[]
                for road in node_dict.in_roads:
                    from_zero = (road["startIntersection"] == node_dict.id) if self.world.RIGHT else (road["endIntersection"] == i.id)
                    for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                        in_lanes.append(road["id"] + "_" + str(n))
                for lane in vehicle_nums.keys():
                    if lane in in_lanes:
                        nvehicles += vehicle_nums[lane]
                        tmp_vehicle.append(vehicle_nums[lane])
                #vehicle_reward.append(-nvehicles)
                tmp_vehicle = np.array(tmp_vehicle)
                vehicle_reward.append(-tmp_vehicle.sum()) #return the average length of a intersection
            vehicle_reward = np.array(vehicle_reward)
            return vehicle_reward
        else:
            raise KeyError("reward_type should in ['pressure', 'vehicleNums']")

    def get_ob(self):
        """
        return: obersavtion of node, observation of edge
        """
        x_obs = [] # num_agents * lane_nums, 
        for i in range(len(self.ob_generators)):
            x_obs.append((self.ob_generators[i][1].generate())/self.vehicle_max)
        # construct edge infomation
        x_obs = np.array(x_obs)
        return x_obs

    def remember(self, ob, phase, action, reward, next_ob, next_phase):
        self.replay_buffer.append((ob, phase, action, reward, next_ob, next_phase))

    def replay(self):
        # need refine later with env
        if self.batch_size > len(self.replay_buffer):
            minibatch = self.replay_buffer
        else:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
        obs = []
        phases = []
        actions = []
        rewards = []
        next_obs = []
        next_phases = []
        for i in range(len(minibatch)):
            obs.append(minibatch[i][0])
            phases.append(minibatch[i][1])
            actions.append(minibatch[i][2])
            rewards.append(minibatch[i][3])
            next_obs.append(minibatch[i][4])
            next_phases.append(minibatch[i][5])
        obs = np.array(obs)
        phases = np.array(phases)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_obs = np.array(next_obs)
        next_phases = np.array(next_phases)
        #obs, phases, edge_obs, actions, rewards, next_obs, next_phases, next_edge_obs = [np.stack(x) for x in np.array(minibatch).T]
        # observations = np.concatenate([next_obs, next_phases],axis=-1)
        my_feed_dict = {self.input_node_state:next_obs, self.input_node_phase:next_phases, self.input_neighbor_id:self.neighbor_id,
                        self.input_node_degree_mask: self.degree_num}
        next_state_value = self.sess.run(self.target_q_value,feed_dict=my_feed_dict)

        next_state_value = np.max(next_state_value,axis=-1) #batch, agents
        target = rewards + self.gamma * next_state_value
        # observations = np.concatenate([obs, phases],axis=-1)
        my_feed_dict = {self.input_node_state:obs, self.input_node_phase:phases, self.input_neighbor_id:self.neighbor_id,
                        self.input_node_degree_mask: self.degree_num, self.input_actions:actions, self.input_target_value:target}
        loss_q,_=self.sess.run([self.q_loss,self.train_op],feed_dict=my_feed_dict)
        loss_q = np.sum(loss_q,axis=-1)
        loss_q = np.mean(loss_q)
        #TODO: print
        variable_names = [v.name for v in tf.trainable_variables()]
        variable_shape = [v.shape for v in tf.trainable_variables()]
        for k,v in zip(variable_names, variable_shape):
            #print("V: ", k)
            #print('shape: ', v)
            pass
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss_q
    
    def update_target_network(self):
        self.sess.run(self.replace_target_op)

    def load_model(self, mdir="model/colight"):
        """
        mdir is the path of model 
        """
        #name = "netlight_agent_{}".format(self.iid)
        # model_name = os.path.join(mdir, name)
        model_name=mdir
        self.algo_saver.restore(self.sess,model_name)
        #self.model.load_weights(model_name)

    def save_model(self, itr, prefix="", mdir="model/colight"):
        name = prefix+"_colight"
        model_name = os.path.join(mdir, name)
        print(model_name)
        self.algo_saver.save(self.sess, model_name, global_step=itr)
        #self.model.save_weights(model_name)