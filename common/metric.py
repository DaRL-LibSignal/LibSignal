import numpy as np


class Metric(object):
    def __init__(lane_metrics, world_metrics,world, agents):
        # must take record of rewards
        self.world = world
        self.agents = agents
        self.decision_num = 0
        self.lane_metrics['rewards'] =  np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        self.lane_metrics.update({k : np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32) for k in self.lane_metrics})
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

    def clear():
        self.lane_metrics['rewards'] =  np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        self.lane_metrics = {k : np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32) for k in self.lane_metrics}
        self.decision_num = 0

    def delay():
        try:
            result = self.lane_metrics['delay']
            return np.sum(result) / (self.decision_num * len(self.world.intersections))
        except KeyError('lane delay is not recorded in lane_metrics, please add it into the list'):
            return None

    def lane_delay():
        try:
            result = self.lane_metrics['delay']
            return result / self.decision_num 
        except KeyError('lane delay is not recorded in lane_metrics, please add it into the list'):
            return None


    def queue():
        try:
            result = self.lane_metrics['queue']
            return np.sum(result) / (self.decision_num * len(self.world.intersections))
        except KeyError('queue in not recorded in lane_matrics, please add it into the list'):
            return None

    def lane_queue():
        try:
            result = self.lane_metrics['queue']
            return result / self.decision_num
        except KeyError('queue in not recorded in lane_matrics, please add it into the list'):
            return None

    def rewards():
        result = self.lane_metrics['rewards']
        return np.sum(result) / self.decision_num
    
    def lane_rewards():
        result = self.lane_metrics['rewards']
        return result / self.decision_num
    
    def throughput():
        return world.get_cur_throughput()
    
    def real_average_travle_time():
        return world.get_average_travel_time()[0]
    
    def plan_average_travle_time():
        return world.get_average_travel_time()[1]
    

    