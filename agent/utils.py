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
        print("t:%s a:%s %s", t, a, greedy_str)
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
        result = (np.array(dict_phase[arr[0]+1])).reshape(-1, len(dict_phase))
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

# def action_convert(action, type='None'):
#     """
#     convert actions format: array<-->number
#     """
#     if type=='array':
#         return np.array([action])
#     elif type == 'num':
#         return action[0]
#     return action


class GetDataSet(torch.utils.data.Dataset):
    # initial, get data
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        '''
        index: obtained by dividing the data according to batchsize
        return: data and the corresponding labels
        '''
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        '''
        return: length of data, facilitate the division of DataLoader
        '''
        return len(self.data)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.layer1 = nn.Sequential(
            OrderedDict([
                (
                    "conv_1",
                    nn.Conv2d(
                        in_channels=1,  # 输入图片的维度
                        out_channels=32,  # 输出图片的维度
                        kernel_size=8,  # 8x8的卷积核，相当于过滤器
                        stride=4,  # 卷积核在图上滑动，每隔一个扫一次
                        padding=3,  # 给图外边补上0
                        bias=False,
                    )),
                ("bn_1", nn.BatchNorm2d(32)),
                ("relu_1", nn.ReLU()),
                ("maxpooling_1", nn.MaxPool2d(kernel_size=2)),
                ("dropout_1", nn.Dropout2d(0.3))
            ]))
        # 第二层卷积
        self.layer2 = nn.Sequential(
            OrderedDict([
                (
                    "conv_2",
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    )),
                ("bn_2", nn.BatchNorm2d(16)),
                ("relu_2", nn.ReLU()),
                ("maxpooling_2", nn.MaxPool2d(kernel_size=2)),
                ("dropout_2", nn.Dropout2d(0.3))
            ]))
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        return out
