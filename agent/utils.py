import numpy as np


def idx2onehot(arr, spices):
    result = np.zeros((arr.shape[0], spices))
    result[range(arr.shape[0]), arr] = 1
    return result
