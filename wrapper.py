import os, sys
import numpy as np
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#os.chdir('test')
import citypb

class Wrapper(object):
    '''
        need to generate:
            X: [T/t, R, C_s+C_d]
            y: [T/t, R]
        where 
            T is the total time span of the flow, 
            t is the length of cycle,
            R is the number of roads
            C_s and C_d are number of channels on the view of Supply and Demand.
    '''
    def __init__(self, roadnet_file, flow_file):
        '''
            
                
        '''
        self.roadnet_file = roadnet_file
        self.flow_file = flow_file
        
        # refer to cbengine/env/CBEngine/envs/CBEngine.py
        # here agent is those intersections with signals
        self.intersections = {}
        self.roads = {}
        self.agents = {}
        self.lane_vehicle_state = {}
        self.log_enable = 1
        self.warning_enable = 1
        self.ui_enable = 1
        self.info_enable = 1
        with open(self.roadnet_file,'r') as f:
            lines = f.readlines()
            cnt = 0
            pre_road = 0
            is_obverse = 0
            for line in lines:
                line = line.rstrip('\n').split(' ')
                if('' in line):
                    line.remove('')
                if(len(line) == 1): ## the notation line
                    if(cnt == 0):
                        self.agent_num = int(line[0])   ## start the intersection segment
                        cnt+=1
                    elif(cnt == 1):
                        self.road_num = int(line[0])*2  ## start the road segment
                        cnt +=1
                    elif(cnt == 2):
                        self.signal_num = int(line[0])  ## start the signal segment
                        cnt+=1
                else:
                    if(cnt == 1):   ## in the intersection segment
                        self.intersections[int(line[2])] = {
                            'latitude':float(line[0]),
                            'longitude':float(line[1]),
                            'have_signal':int(line[3]),
                            'end_roads':[],
                            'start_roads':[]
                        }
                    elif(cnt == 2): ## in the road segment
                        if(len(line)!=8):
                            road_id = pre_road[is_obverse]
                            self.roads[road_id]['lanes'] = {}
                            for i in range(self.roads[road_id]['num_lanes']):
                                self.roads[road_id]['lanes'][road_id*100+i] = list(map(int,line[i*3:i*3+3]))
                                self.lane_vehicle_state[road_id*100+i] = set()
                            is_obverse ^= 1
                        else:
                            self.roads[int(line[-2])]={
                                'start_inter':int(line[0]),
                                'end_inter':int(line[1]),
                                'length':float(line[2]),
                                'speed_limit':float(line[3]),
                                'num_lanes':int(line[4]),
                                'inverse_road':int(line[-1])
                            }
                            self.roads[int(line[-1])] = {
                                'start_inter': int(line[1]),
                                'end_inter': int(line[0]),
                                'length': float(line[2]),
                                'speed_limit': float(line[3]),
                                'num_lanes': int(line[5]),
                                'inverse_road':int(line[-2])
                            }
                            self.intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                            self.intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                            self.intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                            self.intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                            pre_road = (int(line[-2]),int(line[-1]))
                    else:
                        # 4 out-roads
                        signal_road_order = list(map(int,line[1:]))
                        now_agent = int(line[0])
                        in_roads = []
                        for road in signal_road_order:
                            if(road != -1):
                                in_roads.append(self.roads[road]['inverse_road'])
                            else:
                                in_roads.append(-1)
                        in_roads += signal_road_order
                        self.agents[now_agent] = in_roads

                        # 4 in-roads
                        # self.agents[int(line[0])] = self.intersections[int(line[0])]['end_roads']
                        # 4 in-roads plus 4 out-roads
                        # self.agents[int(line[0])] += self.intersections[int(line[0])]['start_roads']
        for agent,agent_roads in self.agents.items():
            self.intersections[agent]['lanes'] = []
            for road in agent_roads:
                ## here we treat road -1 have 3 lanes
                if(road == -1):
                    for i in range(3):
                        self.intersections[agent]['lanes'].append(-1)
                else:
                    for lane in self.roads[road]['lanes'].keys():
                        self.intersections[agent]['lanes'].append(lane)

        # form a dict between road_id and index
        self.idx_dict = {}
        self.id_dict = {}
        self.mask_idx_dict = {}
        self.mask_id_dict = {}
        mask_road_idx = 0
        for idx, road_id in enumerate(self.roads.keys()):
            self.idx_dict[road_id] = idx
            self.id_dict[idx] = road_id
            if len(self.intersections[self.roads[road_id]['start_inter']]['end_roads']) > 1 and\
                len(self.intersections[self.roads[road_id]['end_inter']]['start_roads']) > 1:
                self.mask_idx_dict[road_id] = mask_road_idx
                self.mask_id_dict[mask_road_idx] = road_id
                mask_road_idx += 1
        self.mask = np.zeros((len(self.idx_dict)))
        for i in range(len(self.idx_dict)):
            if self.id_dict[i] in self.mask_idx_dict.keys():
                self.mask[i] = 1
        print("Road Number: {}".format(len(self.idx_dict)))
        print("Option Action Space Length: {}".format(self.mask.sum()))


        # load flow file and obtain the max time length T
        self.time = 0
        with open(self.flow_file,'r') as f:
            lines = f.readlines()
            flow_num = int(lines[0].rstrip('\n').split(' ')[0])
            for flow_idx in range(flow_num):
                line = lines[flow_idx * 3 + 1].rstrip('\n').split(' ')
                if '' in line:
                    line.remove('')
                if len(line) == 3 and int(line[1]) > self.time:
                    self.time = int(line[1])
        print("time:{}".format(self.time))
        # build supply graph
        '''
            Supply graph:
            -Shape: [R, C_s], here we have C_s = 2
            -Feature: 
                [:, 0]: the length of the road
                [:, 1]: the speed limit of the road
                [:, 2]: the num of lanes
        '''
        self.supply_graph = np.zeros((len(self.roads), 3))
        # note that the id should start from 1
        for road_id, road in self.roads.items():
            try:
                self.supply_graph[self.idx_dict[road_id], 0] = road['length']
                self.supply_graph[self.idx_dict[road_id], 1] = road['speed_limit']
                self.supply_graph[self.idx_dict[road_id], 2] = road['num_lanes']
            except:
                pass

        self.n = len(self.idx_dict)


    def test(self):
        warm_up_time = 3600
        engine = citypb.Engine(os.path.join(os.getcwd(), 'configs/openengin1x1.cfg'), 12)
        start_time = time.time()
        for step in range(warm_up_time):
            for intersection in self.intersections.keys():
                engine.set_ttl_phase(intersection, (int(engine.get_current_time()) // 30) % 4 + 1)
            engine.get_lane_vehicles()
            engine.next_step()
            print("t: {}, v: {}".format(step, engine.get_vehicle_count()))
        end_time = time.time()
        print('Runtime: ', end_time - start_time)

        