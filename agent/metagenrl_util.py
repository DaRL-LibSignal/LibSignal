import numpy as np
import scipy.signal
import tensorflow as tf


class DotDict(dict):
    """
    A helper dict that can be accessed using the property notation
    """
    __setattr__ = dict.__setitem__

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            return None


class ReplaceVariableManager:
    """
    This redirect get_variable calls to use the variables from a pre-specified `replace_dict`.
    This is useful to use tensors instead of variables, allowing for easy second order gradients.
    """

    def __init__(self):
        self.replace_dict = None

    def __call__(self, getter, name, *args, **kwargs):
        if self.replace_dict is not None:
            return self.replace_dict[name]
        return getter(name, *args, **kwargs)


def flat(tensors):
    """
    Flatten the given list of tensors into a single vector tensor

    :param tensors: list of tensors
    :return: single tensor with single axis
    """
    return tf.concat([tf.reshape(v, [-1]) for v in tensors if v is not None], axis=0)


def reverse_flat(tensor, shapes):
    """
    Extract back all tensors that were lost when applying utils.flat

    :param tensor: flattened tensor
    :param shapes: shapes of all tensors to be extracted
    :return: list of tensors with the given shapes
    """
    return [tf.reshape(t, shape) for t, shape
            in zip(tf.split(tensor, [shape.num_elements() for shape in shapes]), shapes)]



def correlation(x, y, sample_axis=0):
    x_mean = tf.reduce_mean(x, sample_axis)
    y_mean = tf.reduce_mean(y, sample_axis)
    return (tf.reduce_sum((x - x_mean) * (y - y_mean), sample_axis)
            / tf.sqrt(tf.reduce_sum(tf.square(x - x_mean), sample_axis)
                      * tf.reduce_sum(tf.square(y - y_mean), sample_axis)))


def z_normalize_online(values, axes):
    mean, variance = tf.nn.moments(values, axes)
    return (values - mean) / tf.sqrt(variance + 1e-8)


def repeat(x, count):
    """
    Repeat `x` `count` times along a newly inserted axis at the end
    """
    tiled = tf.tile(x[..., tf.newaxis], [1] * x.shape.ndims + [count])
    return tiled


def discounted_cumsum(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[..., ::-1])[..., ::-1].astype(np.float32)


def tf_discounted_cumsum(t, gamma):
    return tf.py_func(lambda x: discounted_cumsum(x, gamma),
                      [t], tf.float32, stateful=False)


def calculate_gae(rewards, terminals, values, discount_factor, lambda_):
    terminals = tf.cast(terminals, tf.float32)
    not_terminals = 1.0 - terminals
    target = rewards + discount_factor * values[:, 1:] * not_terminals + terminals
    td_residual = target - values[:, :-1]
    advantage = tf_discounted_cumsum(td_residual, discount_factor * lambda_)
    advantage.set_shape(rewards.shape)
    return advantage


def discounted_cumsum_v2(t, gamma):
    """
    Calculates the cumsum in reverse along axis 1 with discount gamma
    :param t: tensor
    :param gamma: discount factor
    :return: tensor of size t with cumsum applied
    """
    t = tf.reverse(tf.transpose(t, perm=[1, 0, 2]), axis=[0])
    r = tf.scan(lambda acc, e: acc * gamma + e, t)
    return tf.transpose(tf.reverse(r, axis=[0]), perm=[1, 0, 2])


def merge_dicts(base: dict, update: dict):
    new = base.copy()
    new.update(update)
    return new


class TensorAdamOptimizer(tf.train.AdamOptimizer):
    """
    Adam Optimizer that can be backpropagated through
    """

    def adapt_gradients(self, grad, var, m=None, v=None, beta1_power=None, beta2_power=None, lr=None):
        if m is None:
            m = self.get_slot(var, "m")
        if v is None:
            v = self.get_slot(var, "v")
        if beta1_power is None or beta2_power is None:
            beta1_power, beta2_power = self._get_beta_accumulators()
        if lr is None:
            lr = self._lr_t

        lr_t = lr * tf.sqrt(1 - beta2_power) / (1 - beta1_power)
        m_t = self._beta1_t * m + (1 - self._beta1_t) * grad
        v_t = self._beta2_t * v + (1 - self._beta2_t) * tf.square(grad)

        # We apply a slightly different rule here by adding epsilon also to sqrt for second-order grad stability
        new_grad = lr_t * m_t / tf.sqrt(v_t + self._epsilon_t)
        new_beta1_power = beta1_power * self._beta1_t
        new_beta2_power = beta2_power * self._beta2_t
        return new_grad, m_t, v_t, new_beta1_power, new_beta2_power


class ConstArray:
    """
    Array-like object that returns a constant for any index
    """

    def __init__(self, value=None):
        self.value = value

    def __getitem__(self, item):
        return self.value


def placeholder(dim=None, time=None, name=None):
    if time is not None:
        shape = (None, time, dim) if dim else (None, time)
    else:
        shape = (None, dim) if dim else (None,)
    return tf.placeholder(dtype=tf.float32, shape=shape, name=name)


def lookup_activation(name):
    if hasattr(tf.nn, name):
        return getattr(tf.nn, name)
    if hasattr(tf, name):
        return getattr(tf, name)
    raise ValueError('Activation not found')


def apply_mixed_activations(x, activations):
    activations = [lookup_activation(a) for a in activations]
    x_s = tf.split(x, len(activations), axis=-1)
    x = tf.concat([a(x_i) for x_i, a in zip(x_s, activations)], axis=-1)
    return x


def get_vars(scope, trainable_only=True):
    if trainable_only:
        return [x for x in tf.trainable_variables() if scope in x.name]
    else:
        return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])