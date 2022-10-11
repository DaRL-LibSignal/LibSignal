import numpy as np


class Metric(object):
    '''
    Register Metric for evaluating model performance. Currently support reward, queue length, delay(approximate or real), throughput and travel time. 
    - Average travel time (travel time): The average time that each vehicle spent on traveling within 
    the network, including waiting time and actual travel time. A smaller travel time means the better performance.
    - Queue length (queue): The average queue length over time, where the queue length at time t 
    is the sum of the number of vehicles waiting on lanes. A smaller queue length means the better performance.
    - Approximated delay (delay): Averaged difference between the current speed of vehicle and the 
    maximum speed limit of this lane over all vehicles, calculated from 1 - (sum_i^n(v_i)/(n*v_max)), where n is the 
    number of vehicles on the lane, v_i is the speed of vehicle i and v_max is the maximum allowed speed. 
    A smaller delay means the better performance.
    - Throughput: Number of vehicles that have finished their trips until current simulation step. A larger
    throughput means the better performance.
    '''
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
        '''
        update
        Recalculate metrics.

        :param rewards: reward name
        :return: None
        '''
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
        '''
        clear
        Reset metrics.

        :param: None
        :return: None
        '''
        self.lane_metrics['rewards'] =  np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        self.lane_metrics = {k : np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32) for k in self.lane_metric_List}
        self.decision_num = 0

    def delay(self):
        '''
        delay
        Calculate vehicle delay.

        :param: None
        :return: real delay or approximate delay
        '''
        # real_delay
        if 'delay' not in self.lane_metrics.keys():
            return self.world.get_real_delay()
        
        # apx_delay
        else:
            try:
                result = self.lane_metrics['delay']
                return np.sum(result) / (self.decision_num * len(self.world.intersections))
            except KeyError:
                print(('apx delay is not recorded in lane_metrics, please add it into the list'))
                return None

    # def lane_delay(self):
    #     try:
    #         result = self.lane_metrics['delay']
    #         return result / self.decision_num 
    #     except KeyError:
    #         print('lane delay is not recorded in lane_metrics, please add it into the list')
    #         return None

    def queue(self):
        '''
        queue
        Calculate total queue length of all lanes.

        :param: None
        :return: total queue length
        '''
        try:
            result = self.lane_metrics['queue']
            return np.sum(result) / (self.decision_num * len(self.world.intersections))
        except KeyError:
            print('queue in not recorded in lane_metrics, please add it into the list')
            return None

    def lane_queue(self):
        '''
        lane_queue
        Calculate average queue length of lanes.

        :param: None
        :return: average queue length of lanes
        '''
        try:
            result = self.lane_metrics['queue']
            return result / self.decision_num
        except KeyError:
            print(('queue in not recorded in lane_metrics, please add it into the list'))
            return None

    def rewards(self):
        '''
        rewards
        Calculate total rewards of all lanes.

        :param: None
        :return: total rewards
        '''
        result = self.lane_metrics['rewards']
        return np.sum(result) / self.decision_num
    
    def lane_rewards(self):
        '''
        lane_rewards
        Calculate average reward of lanes.

        :param: None
        :return: average reward of lanes
        '''
        result = self.lane_metrics['rewards']
        return result / self.decision_num
    
    def throughput(self):
        '''
        throughput
        Calculate throughput.

        :param: None
        :return: current throughput
        '''
        return self.world.get_cur_throughput()
    
    def real_average_travel_time(self):
        '''
        real_average_travel_time
        Calculate average travel time.

        :param: None
        :return: average_travel_time
        '''
        return self.world.get_average_travel_time()
    

    
