import numpy as np


def idx2onehot(arr, spices):
    result = np.zeros((arr.shape[0], spices))
    result[range(arr.shape[0]), arr] = 1
    return result

def action_convert(action, type='None'):
    """
    convert actions format: array<-->number
    """
    if type=='array':
        return np.array([action])
    elif type == 'num':
        return action[0]
    return action
