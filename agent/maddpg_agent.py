from . import RLAgent
import random
import numpy as np
import tensorflow as tf
from . import maddpg_agent_util as U
import tensorflow.contrib.layers as layers

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None, env_index = 0):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [U.make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func_{}".format(env_index), num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func_{}".format(env_index)))

        # wrap parameters in distribution
        # TODO: LOGITS AND SMAPLE
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        test_1 = act_pd.flatparam()
        test_2 = p
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        # TODO: should be onehot, justify here
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func_{}".format(env_index), reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func_{}".format(env_index), num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func_{}".format(env_index)))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64,
            env_index = 0):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [U.make_pdtype(act_space) for act_space in act_space_n]
        # TODO: check full observation dimension
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
            # TODO: notice here
        q = q_func(q_input, 1, scope="q_func_{}".format(env_index), num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func_{}".format(env_index)))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss  # + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        # TODO: see target_ph is here
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func_{}".format(env_index), num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func_{}".format(env_index)))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, args, iid, local_q_func=False):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid

        self.ob_length = ob_generator.ob_length

        self.ob_shape = (self.ob_length,)

        self.last_action = 0

        self.args = args
        self.local_q_func = local_q_func

    def build_model(self, obs_shape_n, act_space_n, agent_index, env_index = 0):
        self.model = mlp_model
        self.name = self.iid
        self.n = len(obs_shape_n)
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=self.model,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.args.lr),
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units,
            env_index = env_index
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=self.model,
            q_func=self.model,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.args.lr),
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units,
            env_index = env_index
        )
        # Create experience buffer
        self.replay_buffer = U.ReplayBuffer(1e4)
        self.max_replay_buffer_len = self.args.batch_size * 25
        self.replay_sample_index = None

    def get_ob(self):
        return self.ob_generator.generate()

    def sample(self):
        return self.action_space.sample()

    def get_reward(self):
        reward = self.reward_generator.generate()[0]
        reward += (self.action == self.last_action) * 2
        self.last_action = self.action
        return reward

    def get_action(self, ob, exploration=False):
        action_prob = self.get_action_prob(ob)
        self.action = np.argmax(action_prob)
        if exploration:
            self.action = np.argmax(action_prob) if random.random() > self.args.epsilon else self.sample()
        return self.action


    def get_action_prob(self, ob):
        return self.act(ob[None])[0]

    # def choose_action(self, obs):
    #     return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 30 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        # print(obs_n.shape)
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))
        self.q_update()
        self.p_update()
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
