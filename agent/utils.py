import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from pfrl import explorers

class SharedEpsGreedy(explorers.LinearDecayEpsilonGreedy):
    def select_action(self, t, greedy_action_func, action_value=None, num_acts=None):
        self.epsilon = self.compute_epsilon(t)
        if num_acts is None:
            fn = self.random_action_func
        else:
            fn = lambda: np.random.randint(num_acts)
        a, greedy = self.select_action_epsilon_greedily(fn, greedy_action_func)
        greedy_str = "greedy" if greedy else "non-greedy"
        # print("t:%s a:%s %s", t, a, greedy_str)
        if num_acts is None:
            return a
        else:
            return a, greedy

    def select_action_epsilon_greedily(self, random_action_func, greedy_action_func):
        if np.random.rand() < self.epsilon:
            return random_action_func(), False
        else:
            return greedy_action_func(), True

def idx2onehot(arr, spices, dict_phase=None):
    if not dict_phase:
        result = np.zeros((arr.shape[0], spices))
        result[range(arr.shape[0]), arr] = 1
    else:
        result = (np.array(dict_phase[arr[0]])).reshape(-1, len(dict_phase))
    return result


def remove_right_lane(ob):
    """
    remove right lane in ob, for some models(eg,frap) do not take right lane into account. 
    """
    if ob.shape[-1] == 8:
        return ob
    elif ob.shape[-1] == 12:
        N = ob[::, 1:3]
        E = ob[::, 4:6]
        S = ob[::, 7:9]
        W = ob[::, 10:12]
        return np.concatenate((E, S, W, N), axis=1)
