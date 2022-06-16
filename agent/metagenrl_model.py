from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib as tfc
import logging

import numpy as np

from . import metagenrl_util as utils
from .metagenrl_util import lookup_activation, apply_mixed_activations

logger = logging.getLogger(__name__)


def mlp(x, output_units, depth, units, activation, use_layernorm, output_activation=None, output_bias=True, **kwargs):
    if output_units:
        depth -= 1
    for i in range(depth):
        if isinstance(activation, list):
            x = tf.layers.dense(x, units=units)
            x = apply_mixed_activations(x, activation)
        else:
            x = tf.layers.dense(x, units=units, activation=activation)
        if use_layernorm:
            use_center = output_bias or i < depth - 1
            x = tfc.layers.layer_norm(x, center=use_center, begin_norm_axis=-1)
    hidden = x
    if output_units:
        out = tf.layers.dense(x, units=output_units, activation=output_activation, use_bias=output_bias)
    else:
        out = hidden
    return utils.DotDict(locals())


def recurrent(x, output_units, depth, units, activation, use_layernorm, initial_state=None, output_activation=None, seq_len=None):
    lstm_cell = tfc.rnn.LayerNormBasicLSTMCell(units, activation=lookup_activation(activation),
                                               layer_norm=use_layernorm)
    inputs = np.repeat(tf.unstack(x, axis=1), depth).tolist()
    outputs, state = tf.nn.static_rnn(lstm_cell, inputs, dtype=tf.float32, initial_state=initial_state,
                                      sequence_length=seq_len)
    hidden = tf.stack(outputs[depth - 1::depth], axis=1)
    out = tf.layers.dense(hidden, output_units, activation=output_activation)
    return utils.DotDict(locals())


class Agent:
    """
    A tensorflow model for the agents with critic and policy (+ target networks)
    """

    def __init__(self, dconfig, obs_shape_n, act_space_n):
        self.obs_dim = obs_shape_n
        self.act_dim = act_space_n
        # self.act_limit = env.action_space.high[0]

        if dconfig.critic_is_recurrent:
            critic_args = [dconfig.critic_depth, dconfig.critic_units, dconfig.critic_rnn_activation,
                           dconfig.critic_layernorm]
            self.critic_func = lambda *args, **kwargs: recurrent(*args, *critic_args, **kwargs)
        else:
            critic_args = [dconfig.critic_depth, dconfig.critic_units, dconfig.critic_activation,
                           dconfig.critic_layernorm]
            self.critic_func = lambda *args, **kwargs: mlp(*args, *critic_args, **kwargs)

        if dconfig.policy_is_recurrent:
            policy_args = [dconfig.policy_depth, dconfig.policy_units, dconfig.policy_rnn_activation,
                           dconfig.policy_layernorm]
            self.policy_func = lambda *args, **kwargs: recurrent(*args, *policy_args, **kwargs)
        else:
            policy_args = [dconfig.policy_depth, dconfig.policy_units, dconfig.policy_activation,
                           dconfig.policy_layernorm]
            self.policy_func = lambda *args, **kwargs: mlp(*args, *policy_args, **kwargs)

        with tf.variable_scope(None, 'agents'):
            self.main = self._create('main')
            self.target = self._create('target')

    def _create(self, scope):
        with tf.variable_scope(scope):
            replace_manager = utils.ReplaceVariableManager()
            return utils.DotDict({
                'critic': tf.make_template('critic', self._critic, True),
                'critic2': tf.make_template('critic2', self._critic, True),
                'policy': tf.make_template('policy', self._policy, True, custom_getter_=replace_manager)
            })

    def _critic(self, x, a, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        value = tf.squeeze(self.critic_func(tf.concat([x, a], axis=-1), 1, **kwargs).out, axis=-1)
        return value

    def _policy(self, x, initial_state=None, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if initial_state is not None:
            initial_state = tf.unstack(initial_state)
            kwargs['initial_state'] = initial_state
        policy = self.policy_func(x, self.act_dim, output_activation=tf.tanh, **kwargs)
        pi = policy.out
        result = {
            'action': pi,
            'hidden': policy.hidden,
            'value': self.main.critic(x, pi),
            'target_value': lambda: self.target.critic(x, pi),
        }
        if hasattr(policy, 'state'):
            result['state'] = policy.state
        return utils.DotDict(result)


class Objective:
    """
    A neural objective function
    """

    def __init__(self, dconfig):
        self.dconfig = dconfig
        self.objective = tf.make_template('objective', self._create_objective, True)

        obj_args = [dconfig.obj_func_depth, dconfig.obj_func_units, dconfig.obj_func_activation,
                    dconfig.obj_func_layernorm]
        self.obj_func = lambda *args, **kwargs: mlp(*args, *obj_args, **kwargs)

        if dconfig.obj_func_input_transform_depth:
            input_transform_kwargs = {'depth': dconfig.obj_func_input_transform_depth,
                          'output_units': 0,
                          'units': dconfig.obj_func_input_transform_units,
                          'activation': dconfig.obj_func_activation,
                          'use_layernorm': dconfig.obj_func_layernorm}
            self.input_transform = lambda *args, **kwargs: mlp(*args, **utils.merge_dicts(input_transform_kwargs, kwargs)).out
        else:
            self.input_transform = None

    def _objective_reward_value_transform(self, values, rewards, terminals, create_summary):
        """
        First transformation on objective inputs
        """
        values = values[..., tf.newaxis]
        normalized_values = utils.z_normalize_online(values, axes=[0, 1])
        normalized_rewards = utils.z_normalize_online(rewards, axes=[0, 1])

        time = tf.tile(tf.range(0, rewards.shape[1].value, dtype=tf.float32)[tf.newaxis, :, tf.newaxis],
                       [tf.shape(values)[0], 1, 1])
        inp = tf.concat([normalized_rewards,
                         time / rewards.shape[1].value,
                         normalized_values[:, 1:] * (1.0 - terminals),
                         normalized_values[:, :-1]], axis=-1)

        if create_summary:
            tf.summary.histogram('obj_input', inp)
        return self.input_transform(inp, use_layernorm=self.dconfig.obj_func_input_transform_layernorm,
                                    output_units=self.dconfig.obj_func_input_transform_out_units)

    def _objective_error_transform(self, inp):
        """
        Takes a vector and transforms it to a bounded scalar error
        """
        use_error_scale = self.dconfig.obj_func_error_scale is not None
        use_error_func = self.dconfig.obj_func_error_func is not None
        error = tf.squeeze(self.obj_func(inp, 1, output_bias=use_error_func).out, axis=-1)

        if use_error_scale:
            error = error * self.dconfig.obj_func_error_scale
        if use_error_func:
            func = self.dconfig.obj_func_error_func
            error = getattr(tf.nn, func, getattr(tf, func))(error)
        if use_error_scale:
            error = error / self.dconfig.obj_func_error_scale

        return error

    def _create_objective(self,trans, seq_len, agent, policy, create_summary=False):
        ftype = self.dconfig.obj_func_type

        if ftype == 'learned-reinforce':
            # Only support entire trajectories and non recurrent critics at the moment
            assert self.dconfig.recurrent_time_steps > 1
            assert not self.dconfig.critic_is_recurrent

            _, rb_action, x2, rewards, terminals = trans
            # TODO can we already compute this in the first pass?
            final_input = x2[:, -1]

            if self.dconfig.policy_is_recurrent:
                # TODO actually we can not just use zero here
                #  because the last observation may not be at the end of an episode
                #  (recurrent version is currently not used)
                final_value = tf.zeros(tf.shape(policy.value)[0])
            else:
                final_value = agent.main.policy(final_input).value

            values = tf.stop_gradient(tf.concat([policy.value, final_value[:, tf.newaxis]], axis=-1))
            obj_action_input = tf.concat([rb_action[..., tf.newaxis], policy.action[..., tf.newaxis]], axis=-1)
            if create_summary:
                tf.summary.histogram('obj_action_input', obj_action_input)
            transformed_actions = tf.reduce_mean(self.input_transform(obj_action_input), axis=-2)
            transformed_other_inputs = self._objective_reward_value_transform(values, rewards, terminals,
                                                                              create_summary)
            rnn_input = tf.unstack(tf.concat([transformed_actions, transformed_other_inputs], axis=-1), axis=1)[::-1]

            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.dconfig.obj_func_lstm_units)
            outputs, _ = tf.nn.static_rnn(lstm_cell, rnn_input, dtype=tf.float32, sequence_length=seq_len)
            outputs = tf.stack(outputs[::-1], axis=1)

            return self._objective_error_transform(outputs)
        elif ftype == 'reinforce':
            # A baseline objective function: off-policy REINFORCE
            assert self.dconfig.recurrent_time_steps > 1
            assert not self.dconfig.policy_is_recurrent
            assert not self.dconfig.critic_is_recurrent
            _, rb_action, x2, rewards, terminals = trans
            # TODO can we already compute this in the first pass?
            final_input = x2[:, -1]
            final_value = agent.main.policy(final_input).value
            values = tf.concat([policy.value, final_value[:, tf.newaxis]], axis=-1)
            gae = utils.calculate_gae(tf.squeeze(rewards, axis=-1), tf.squeeze(terminals, axis=-1),
                                      values, self.dconfig.discount_factor, self.dconfig.gae_factor)
            error = tf.reduce_mean((rb_action - policy.action) ** 2, axis=-1)
            return error * gae * tf.get_variable('factor', [], tf.float32, initializer=tf.ones_initializer())
        else:
            raise ValueError(f'Invalid objective function type:{ftype}')

    def future_policy_value(self, x, a, trans, seq_len, seq_mask, agent, opt, create_summary=False):
        """
        Computes the value of a policy according to the critic when updated using the objective function
        :param x: observations
        :param a: actions
        :param trans: entire tuple of transition (s_t, a_t, r_t, d_t, s_{t+1})
        :param seq_len: Length of trajectories
        :param seq_mask: Binary mask of trajectories
        :param agent: agents to compute value for
        :param opt: optimizer to use for the policy update
        :param create_summary: whether to create summary ops
        :return: tensor of batched future policy value
        """
        with tf.variable_scope('future_policy_value'):
            policy = agent.main.policy
            policy_vars = policy.trainable_variables
            # The replace manager can replace the policy variables with updated variables
            replace_manager = policy.variable_scope.custom_getter

            use_adam = self.dconfig.obj_func_second_order_adam
            step_size = self.dconfig.obj_func_second_order_stepsize
            step_count = self.dconfig.obj_func_second_order_steps + 1
            batch_size = self.dconfig.buffer_sample_size

            # Split tensors according to number of inner gradient descent steps
            x_s = tf.split(x, step_count, axis=0)
            a_s = tf.split(a, step_count, axis=0)
            if seq_len is not None:
                seq_len_s = tf.split(seq_len, step_count, axis=0)
                seq_mask_s = tf.split(seq_mask, step_count, axis=0)
            else:
                seq_len_s = utils.ConstArray()
                seq_mask_s = utils.ConstArray(seq_mask)
            trans_s = list(zip(*(tf.split(e, step_count, axis=0) for e in trans)))

            objective_val = None
            policy_grads = None
            opt_args_dict = {}
            current_vars = policy_vars
            var_names = [var.op.name for var in policy_vars]
            for i in range(step_count - 1):
                # Run policy
                policy_result = policy(x_s[i], seq_len=seq_len_s[i])
                # Run objective
                objective_val = self.objective(x_s[i], a_s[i], trans_s[i], seq_len_s[i], seq_mask_s[i], agent,
                                               policy_result, create_summary)
                # Compute policy gradients
                policy_grads = tf.gradients(objective_val * seq_mask_s[i], current_vars)

                if use_adam:
                    def grad_transform(grad, var, var_name):
                        if var_name in opt_args_dict:
                            opt_args = opt_args_dict[var_name]
                        else:
                            opt_args = []
                        new_grad, *opt_args = opt.adapt_gradients(grad, var, *opt_args, lr=step_size)
                        opt_args_dict[var_name] = opt_args
                        return new_grad
                else:
                    def grad_transform(grad, *args):
                        return step_size * grad

                # Use adam or vanilla SGD for inner gradient step
                transformed_grads = [grad_transform(grad, var, var_name)
                                     for grad, var, var_name in zip(policy_grads, current_vars, var_names)]

                one_step_updated_policy_vars = [var - grad for var, grad in zip(current_vars, transformed_grads)]
                one_step_updated_policy_vars_dict = OrderedDict(zip(var_names, one_step_updated_policy_vars))

#               # Updates replace manager to run policy with updated variables in the next loop iteration
                replace_manager.replace_dict = one_step_updated_policy_vars_dict
                current_vars = one_step_updated_policy_vars

            # Run policy with final parameters
            future_policy = policy(x, seq_len=seq_len)
            replace_manager.replace_dict = None
            # Estimate the final policy value
            future_policy_value = agent.main.critic(x, future_policy.action) * seq_mask

            if create_summary:
                orig_policy = policy(x_s[-1], seq_len=seq_len_s[-1])
                partial_future_policy_value = future_policy_value[-batch_size:]
                tf.summary.histogram('objective_value', objective_val)
                tf.summary.histogram('policy_grads', utils.flat(policy_grads))
                tf.summary.histogram('policy_value', orig_policy.value)
                tf.summary.histogram('future_policy_value', partial_future_policy_value)
                tf.summary.histogram('policy_value_gain', partial_future_policy_value - orig_policy.value)

                sample_axis = [0, 1] if self.dconfig.recurrent_time_steps > 1 else 0
                cor = utils.correlation(-orig_policy.value, objective_val, sample_axis)
                tf.summary.scalar('objective_critic_correlation', tf.squeeze(cor))

                grad, = tf.gradients(objective_val, policy_result.value)
                if grad is not None:
                    tf.summary.histogram('objective_critic_grads', grad)

        return future_policy_value

    @property
    def variables(self):
        return self.objective.trainable_variables

    def set_variables(self, sess, values):
        for var, val in zip(self.variables, values):
            var: tf.Variable
            var.load(val, sess)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size, discount_factor):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.discount_factor = discount_factor
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.episode_markers = [0]  # Can't use dequeue here due random access sampling
        self.ptr, self.size, self.max_size = 0, 0, size

    def restore(self, other: 'ReplayBuffer'):
        self.obs1_buf = other.obs1_buf
        self.obs2_buf = other.obs2_buf
        self.acts_buf = other.acts_buf
        self.rews_buf = other.rews_buf
        self.done_buf = other.done_buf
        self.episode_markers = other.episode_markers
        self.ptr = other.ptr
        self.size = other.size
        self.max_size = other.max_size

    def store(self, obs, act, rew, next_obs, done):
        if self.done_buf[self.ptr]:
            del self.episode_markers[0]

        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        if done:
            self.episode_markers.append(self.ptr)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_time_batch(self, time, batch_size):
        eps_idxs = np.random.randint(0, len(self.episode_markers) - 1, size=batch_size)
        eps = np.array([self.episode_markers[i] for i in eps_idxs])
        eps_lens = np.array([self.episode_markers[i + 1] - self.episode_markers[i] for i in eps_idxs])
        offsets = np.array([np.random.randint(0, max(eps_len - time + 1, 1)) for eps_len in eps_lens])

        lens = np.minimum(eps_lens, time)
        idxs = eps + offsets

        def create(buf, use_ones=False):
            shape = (batch_size, time) + buf.shape[1:]
            out = np.ones(shape) if use_ones else np.zeros(shape)
            for i, (idx, len_) in enumerate(zip(idxs, lens)):
                out[i, :len_] = buf[idx:idx + len_]
            return out

        return dict(obs1=create(self.obs1_buf),
                    obs2=create(self.obs2_buf),
                    acts=create(self.acts_buf),
                    rews=create(self.rews_buf),
                    done=create(self.done_buf),
                    lens=lens)

    def create_dataset(self, batch_size, time=None):
        """
        Create a tf dataset from this replay buffer
        :param batch_size: the mini batch size to use
        :param time: whether to sample trajectories of length `time` or single transitions
        :return: a tf dataset
        """
        output_types = dict(
            obs1=tf.float32,
            obs2=tf.float32,
            acts=tf.float32,
            rews=tf.float32,
            done=tf.float32,
        )
        if time is None:
            def _generator():
                while True:
                    yield self.sample_batch(batch_size)
            output_shapes = dict(
                obs1=[None, self.obs_dim],
                obs2=[None, self.obs_dim],
                acts=[None, self.act_dim],
                rews=[None],
                done=[None],
            )
        else:
            def _generator():
                while True:
                    yield self.sample_time_batch(time, batch_size)
            output_types['lens'] = tf.int32
            output_shapes = dict(
                obs1=[None, time, self.obs_dim],
                obs2=[None, time, self.obs_dim],
                acts=[None, time, self.act_dim],
                rews=[None, time],
                done=[None, time],
                lens=[None]
            )
        dataset = tf.data.Dataset.from_generator(_generator, output_types, output_shapes)
        dataset = dataset.prefetch(3)
        return dataset
