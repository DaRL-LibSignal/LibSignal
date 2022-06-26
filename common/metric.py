import numpy as np


class Metric(object):
    def __init__(self, lane_metrics, world_metrics, world, agents):
        # must take record of rewards
        self.world = world
        self.agents = agents
        self.decision_num = 0
        self.lane_metric_List = lane_metrics
        self.lane_metrics = dict()
        self.lane_metrics['rewards'] =  np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        self.lane_metrics.update({k : np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32) for k in self.lane_metric_List})
        self.world_metrics = world_metrics

    def update(self, rewards=None):
        if rewards is not None:
            self.lane_metrics['rewards'] += rewards.flatten()
        if 'delay' in self.lane_metrics.keys():
            self.lane_metrics['delay'] += (np.stack(np.array(
                [ag.get_delay() for ag in self.agents], dtype=np.float32))).flatten()
        if 'queue' in self.lane_metrics.keys():
            self.lane_metrics['queue'] += (np.stack(np.array(
                [ag.get_queue() for ag in self.agents], dtype=np.float32))).flatten()
        self.decision_num += 1

    def clear(self):
        self.lane_metrics['rewards'] =  np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        self.lane_metrics = {k : np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32) for k in self.lane_metric_List}
        self.decision_num = 0

    def delay(self):
        try:
            result = self.lane_metrics['delay']
            return np.sum(result) / (self.decision_num * len(self.world.intersections))
        except KeyError:
            print(('lane delay is not recorded in lane_metrics, please add it into the list'))
            return None

    def lane_delay(self):
        try:
            result = self.lane_metrics['delay']
            return result / self.decision_num 
        except KeyError:
            print('lane delay is not recorded in lane_metrics, please add it into the list')
            return None


    def queue(self):
        try:
            result = self.lane_metrics['queue']
            return np.sum(result) / (self.decision_num * len(self.world.intersections))
        except KeyError:
            print('queue in not recorded in lane_metrics, please add it into the list')
            return None

    def lane_queue(self):
        try:
            result = self.lane_metrics['queue']
            return result / self.decision_num
        except KeyError:
            print(('queue in not recorded in lane_metrics, please add it into the list'))
            return None

    def rewards(self):
        result = self.lane_metrics['rewards']
        return np.sum(result) / self.decision_num
    
    def lane_rewards(self):
        result = self.lane_metrics['rewards']
        return result / self.decision_num
    
    def throughput(self):
        return self.world.get_cur_throughput()
    
    def real_average_travel_time(self):
        return self.world.get_average_travel_time()[0]
    
    def plan_average_travel_time(self):
        return self.world.get_average_travel_time()[1]
    

    