import json
import os.path as osp
import citypb
from common.registry import Registry

import time
import os
from math import atan2, pi
import random
import gc


def _get_direction(road):
    x = road[1]['x'] - road[0]['x']
    y = road[1]['y'] - road[0]['y']
    tmp = atan2(x, y)
    return tmp if tmp >= 0 else (tmp + 2 * pi)


class Intersection(object):
    def __init__(self, intersection, world):
        self.id = intersection['id']
        self.world = world
        self.eng = self.world.eng
        # TODO: check if its from North
        #incoming and outgoing roads of each intersection, clock-wise order from North
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = None
        self.in_roads = None

        # TODO: how to control phase and links in cbengine?
        # links and phase information of each intersection
        self.current_phase = 0
        self.roadlinks = []
        self.lanelinks_of_roadlink = []
        self.startlanes = []
        self.lanelinks = []
        self.phase_available_roadlinks = []
        self.phase_available_lanelinks = []
        self.phase_available_startlanes = []

        # TODO: yellow phase control in cbengine?

        # road processing directly by intersection
        for road_id in intersection['roads']:
            self.roads.append(road_id)
            # use start_roads to judge
            self.outs.append(True if road_id in intersection['start_roads'] else False)
            self.directions.append(intersection['direction'][road_id])
        # process road_lane_mapping here
        self.road_lane_mapping = dict()
        for road in intersection['road_lane_mapping'].keys():
            self.road_lane_mapping[road] = []
            for lane in intersection['road_lane_mapping'][road]:
                self.road_lane_mapping[road].append(lane)

        self._sort_roads()
        # lanes of this intersection
        self.lanes = []
        for road in self.road_lane_mapping:
            for lane in self.road_lane_mapping[road]:
                self.lanes.append(lane)
        # TODO: set phase and yellow phase in the future
        self.phases = [i for i in range(8)]
        self.current_phase = 0
        self.current_phase_time = 0
        self.yellow_phase_time = 0

        # set vehicle info on lanes. should be {in_time, speed, accumulated_wait_time, }
        self.vehicles_cur = {lane: dict() for lane in self.lanes}
        # should be {in_time, out_time}
        self.vehicles = {lane: dict() for lane in self.lanes}
        # control vehicle info on each road since it has lane change
        self.vehicles_cur_road = {road: dict() for road in self.roads}
        self.vehicles_road = {road: dict() for road in self.roads}

        self.full_observation = {lane: dict() for lane in self.lanes}

    def _sort_roads(self):
        order = sorted(range(len(self.roads)),
                       key=lambda i: (self.directions[i],
                                      self.outs[i] if self.world.RIGHT else not self.outs[i]))
        # TODO: check order [2,1,0] if self.RIGHT
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]

    def reset():
        self.current_phase = 0
        self.current_phase_time = 0
        self.vehicles_cur = {lane: dict() for lane in self.lanes}
        self.vehicles = {lane: dict() for lane in self.lanes}
        self.vehicles_cur_road = {road: dict() for road in self.roads}
        self.vehicles_road = {road: dict() for road in self.roads}
        self.full_observation = {lane: dict() for lane in self.lanes}

    def observe(self):
        # TODO: DOUBLE CHECK IF OUT LANE COUNT?
        speed = self.world.eng.get_vehicle_speed()
        # for debug
        """
        print('\n')
        print('lane_vehicle: ', self.eng.get_lane_vehicles())
        """
        for lane in self.lanes:
            lane_vehicles = [v 
                for v in self.eng.get_lane_vehicles().get(lane, [])] # basic information returned from simulator
            # process each vehicles on this lane
            for v in lane_vehicles:
                if v in self.vehicles_cur[lane].keys():
                    self.vehicles_cur[lane][v]['speed'] = speed[v]
                    if self.vehicles_cur[lane][v]['speed'] == 0:
                        self.vehicles_cur[lane][v]['wait_time'] += 1
                elif v not in self.vehicles_cur_road[int(str(lane)[:-2])]:
                    # TODO: new vehicle. since start speed is set to be 0, modifiy it
                    self.vehicles_cur[lane][v] = {'speed': -1, 'wait_time': -1,
                     'start_time': self.world.eng.get_current_time()}
                    self.vehicles_cur_road[int(str(lane)[:-2])].update({v: lane})
                else:
                    assert v not in self.vehicles_cur[lane].keys()
                    road = lane // 100
                    for para_lane in self.road_lane_mapping[road]:
                        if v in self.vehicles_cur[para_lane]:
                            out = False
                            # change lanes
                            prev_lane = self.vehicles_cur_road[int(str(lane)[:-2])][v]
                            self.vehicles_cur[lane][v] = self.vehicles_cur[para_lane].pop(v)
                            break
        for lane in self.lanes:
            lane_mearsures = {'lane_waiting_time_count': 0, 'lane_waiting_count': 0,
            'lane_count': 0, 'queue_length': 0}
            lane_vehicles = [v 
                for v in self.world.eng.get_lane_vehicles().get(lane, [])]
            out = self.vehicles_cur[lane].keys() - lane_vehicles
            for v in out:
                self.vehicles[lane].update({v: {'start_time': self.vehicles_cur[lane][v]['start_time'],
                 'end_time': self.world.eng.get_current_time()}})
                 # could combine this two step
                self.vehicles_cur[lane].pop(v)
                self.vehicles_cur_road[int(str(lane)[:-2])].pop(v)
                # TODO: CHECKOU HERE WHY NOT POPED
            # process information of this lane based on vehicles states
            
            for k, v in self.vehicles_cur[lane].items():
                if v['speed'] == 0:
                    lane_mearsures['lane_waiting_count'] += 1
                    # TODO: talk about it later
                if v['wait_time'] > 0:
                    lane_mearsures['lane_waiting_time_count'] += v['wait_time']
                    # TODO: add queue length later
                lane_mearsures['lane_count'] += 1
            self.full_observation[lane] = lane_mearsures
        test = 0
        """
        print('\n')
        print('vehicle: ')
        for lane in self.vehicles.keys():
            if self.vehicles[lane].keys():
                print(lane, ': ' , self.vehicles[lane].keys())
        print('vehicle: ', self.vehicles)
        print('\n')
        print('vehicle_cur : ')
        for lane in self.vehicles_cur.keys():
            if self.vehicles_cur[lane].keys():
                print(lane, ': ' , self.vehicles_cur[lane].keys())
        """

    def psedo_step(self, action=None):
        # No observation here. Update after step() finished
        if action != self.current_phase:
            self.current_phase = action
            self.current_phase_time = 0
        self.current_phase_time += 1
        self.world.eng.set_ttl_phase(self.id, action)


@Registry.register_world('openengine')
class World(object):
    """
    Create a Citypb engine and maintain infromations about Citypb world
    """
    def __init__(self, citypb_config, thread_num):
        print("building world from cbengine...")
        self.cfg_file = citypb_config
        self.eng = citypb.Engine(citypb_config, thread_num)

        with open(citypb_config) as f:
            citypb_config = f.readlines()
        # process road_file
        flag = 0
        for line in citypb_config:
            if ("interval" in line):
                interval = line.split('=')[1].strip(' ') 
                flag += 1
            if ("road_file_addr" in line):
                roadnet_path = line.split(':')[1].strip('\n').strip(' ')
                flag += 1
                break
        assert flag == 2, 'please provide interval and road_file_addr in cfg file'


        self.roadnet = self._get_roadnet(roadnet_path)
        self.RIGHT = True  # TODO: provide right driver
        self.interval = interval
        # will be generated in _get_roadnet function
        # Intersection(self.roadnet["intersections"][i]
        self.intersections = [Intersection(self.roadnet["intersections"][i], self)
            for i in self.roadnet["intersections"] if not self.roadnet["intersections"][i]["virtual"]]
        print('intersection created')
        self.id2intersection = dict()
        for inter in self.intersections:
            self.id2intersection[inter.id] = inter
        self.intersection_ids = [i.id for i in self.intersections]
        # set all roads and all lanes
        self.all_lanes = []
        self.all_roads = []
        for itsec in self.intersections:
            for road in itsec.road_lane_mapping.keys():
                if itsec.road_lane_mapping[road] and road not in self.all_roads:
                    # append road name into all_roads if road exists
                    self.all_roads.append(road)
                    for lane in itsec.road_lane_mapping[road]:
                        if lane not in self.all_lanes:
                            self.all_lanes.append(lane)
        # lane maxSpeed. different lanes share same road max speed
        self.lane_maxSpeed = {}
        for road in self.roadnet['roads'].keys():
            for lane in self.roadnet['roads'][road]['lanes']:
                self.lane_maxSpeed[lane] = self.roadnet['roads'][road]['maxSpeed']
        print('road parsed')

        self.info_functions = {
            "vehicles": self.get_vehicles,
            "lane_count": self.get_lane_vehicle_count,
            "lane_waiting_count": self.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.get_lane_vehicles,
            "time": self.eng.get_current_time,
            "vehicle_distance": None,
            "pressure": self.get_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "vehicle_trajectory": None,
            "history_vehicles": None,
            "phase": self.get_cur_phase,
            "throughput": None,
            "average_travel_time": None
        }
        self.fns = []
        self.info = {}
        # subscibe it to process vehicles on each lane throught intersection objects
        self.subscribe('lane_vehicles')
        self._update_infos()


    def _get_roadnet(self, citypb_path):
        """
        generate roadnet dictionary based on providec configuration file
        functions borrowed form openengine CBEngine.py
        Details:
        collect roadnet informations.
        {1-'intersections'-(len=N_intersections):
            {key='id': name of the intersection,
             11-'id': id,
             12-'point': 121: {'x', 'y'}(intersection at this position),
             13-'roads'(len=N_roads controled by this intersection): [(id)],
             14-'start_roads'(len(N_start_roads)): [(id)],
             15-'end_roads'(len=N_end_roads): [(id)],
             16-'direction': degree to x,
             17-'road_lane_mapping': 
                {key='id_road'(len=N_lanes_in_road): [
                    {key='id_lane': name of lane based on road id
                        161-'type': [x, x, x]]
                 }
             18-'direction': {'road_id': (angle between road and x axis)}
             19-'virtual': bool
             1A*-'roadLinks'(len=N_road links): 
                {1A1-'type': diriction type(go_straight, turn_left, turn_right, turn_U),  # TODO: check turn_u
                 1A2-'startRoad': start road name,
                 1A3-'endRoad': end road name,
                 1A4-'direction': int(same as type)
                 1A5-'laneLinks(len-N_lane links of road): 
                 },
             1B*-'trafficLight: 
                {1B1-'roadLinkIndices'(len=N_road links): [],
                 1B2-'lightphases'(len=N_phases): {1B11-'time': int(time long),
                                                    1B12-'availableRoadLinks'(len=N_working_roads): []
                                                    }
                 },
             },
         2-'roads'-(len=N_roads ): 
            {key='id': name of the road,
             21-'id': id
             22-'startIntersection: road start,
             23-'endIntersection: road end,
             24-'points': [241: {'x', 'y'}(start pos), 242: {'x', 'y'}(end pos)],
             25-'length': int,
             26-'maxSpeed': int,
             27-'Nlane': int,
             28-'lanes'(N_lanes in this road): {key=road_id+rank: [x, x, x](type)}
             29-'inverse': int
             }
         }
         TODO: expalain - need points to rank features

        """
        result = dict({'intersections': {}, 'roads': {}})
        with open(citypb_path, 'r') as f:
            lines = f.readlines()
            cnt = 0
            pre_road = 0
            is_obverse = 0
            for line in lines:
                # preprocessing line formation
                line = line.rstrip('\n').split(' ')
                if ('' in line):
                    line.remove('')
                if (len(line) == 1):  # this is notation 
                    if cnt == 0:
                        agent_num = int(line[0])  # start intersection segment
                        cnt += 1
                    elif cnt == 1:
                        road_num = int(line[0]) * 2  # start road segment
                        cnt += 1
                    elif cnt == 2:
                        signal_num = int(line[0])  # start signal segment
                        cnt += 1
                else:
                    if cnt == 1:  # intersection processing
                        # id, point, width, roads, roadLinks, trafficLight, virtual
                        result['intersections'].update({
                            # No width, No detailed information about road links
                            int(line[2]): {'point': {'x': float(line[0]), 'y': float(line[1])}, 'direction': {}, 'id': int(line[2]),
                            'virtual': bool(abs(1 -int(line[3]))), 'start_roads': [], 'end_roads': [], 'roads': [], 'road_lane_mapping': {}}
                            # TODO: laneLinks, roadLinks maybe important
                            }
                        )
                    if cnt == 2: # road processing
                        if len(line) != 8:
                            road_id = pre_road[is_obverse]
                            # add road details in the last line generated road 
                            result['roads'][road_id]['lanes'] = {}
                            for i in range(result['roads'][road_id]['Nlane']):
  #!!!                          # TODO: two lanes ? 
                                result['roads'][road_id]['lanes'][road_id*100+i] = list(map(int,line[i:i*3+3]))
                            is_obverse ^= 1
                        else:
                            # road is added here, then modified by code above
                            result['roads'].update({
                                # TODO: make sure there all lane in the same road shares the same maxspeed
                                int(line[-2]): {'startIntersection': int(line[0]), 'endIntersection': int(line[1]),
                                                'points': [{'x': result['intersections'][int(line[0])]['point']['x'],
                                                             'y': result['intersections'][int(line[0])]['point']['y']
                                                             },
                                                            {'x': result['intersections'][int(line[1])]['point']['x'],
                                                             'y': result['intersections'][int(line[1])]['point']['y']
                                                            }],
                                                'length': float(line[2]), 'maxSpeed': float(line[3]), 'lanes': [],
                                                'inverse': int(line[-1]), 'Nlane': int(line[4])}
                            })
                            # revert to add road from reversed direction
                            # TODO: maybe revised to support one-way by the Engine provider in the future
                            result['roads'].update({
                                int(line[-1]): {'startIntersection': int(line[1]), 'endIntersection': int(line[0]),
                                                'points': [{'x': result['intersections'][int(line[1])]['point']['x'],
                                                             'y': result['intersections'][int(line[1])]['point']['y']
                                                             },
                                                            {'x': result['intersections'][int(line[0])]['point']['x'],
                                                             'y': result['intersections'][int(line[0])]['point']['y']
                                                            }],
                                                'length': float(line[2]), 'maxSpeed': float(line[3]), 'lanes' : [],
                                                'Nlane': int(line[5]), 'inverse': int(line[-2])}
                            })
                            result['intersections'][int(line[0])]['end_roads'].append(int(line[-1]))
                            result['intersections'][int(line[0])]['roads'].append(int(line[-1]))
                            result['intersections'][int(line[1])]['end_roads'].append(int(line[-2]))
                            result['intersections'][int(line[1])]['roads'].append(int(line[-2]))
                            result['intersections'][int(line[0])]['start_roads'].append(int(line[-2]))
                            result['intersections'][int(line[0])]['roads'].append(int(line[-2]))
                            result['intersections'][int(line[1])]['start_roads'].append(int(line[-1]))
                            result['intersections'][int(line[1])]['roads'].append(int(line[-1]))
                            pre_road = (int(line[-2]),int(line[-1]))

                    else:
                        pass
                        """
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
                        """
        print('intersection processed')
        # TODO: For restoring information of each roads' lane information to intersections, check if its right.
        for road_idx in result['roads']:
            road = result['roads'][road_idx]
            direction = _get_direction(road['points'])
            lane_in_road = road['lanes']
                # append road's lane information to start intersection
            road_lane_mapping = {}
            for lane_id in lane_in_road:
                road_lane_mapping.update({lane_id: lane_in_road[lane_id]})
            inter_1 = road['startIntersection']
            result['intersections'][inter_1]['direction'].update({road_idx: direction})
            result['intersections'][inter_1]['road_lane_mapping'].update({road_idx: road_lane_mapping})
            # append road's lane information to end intersection
            road_lane_mapping = {}
            for lane_id in lane_in_road:
                road_lane_mapping.update({lane_id: lane_in_road[lane_id]})
            inter_2 = road['endIntersection']
            result['intersections'][inter_2]['direction'].update({road_idx: direction})
            result['intersections'][inter_2]['road_lane_mapping'].update({road_idx: road_lane_mapping})
        print('roads processed')
        return result
    
    def step(self, action=None):
        if action is not None:
            for i, inter in enumerate(self.intersections):
                # set phase within intersections
                inter.psedo_step(action[i])
        # TODO: so now we don't process delay and queue. just some basic metric first
            self.eng.next_step()
            # update lane information of each intersection
            self._update_infos()
            for inter in self.intersections:
                inter.observe()
        else:
            raise Exception('provide action in RL or need some spefic design for non-RL agents')

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if fn not in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception(f'Info function {fn} not implemented')
    
    def reset(self):
        del self.eng
        gc.collect()
        self.eng = citypb.Engine(self.cfg_file, 12)
        self.intersections = [Intersection(self.roadnet["intersections"][i], self)
            for i in self.roadnet["intersections"] if not self.roadnet["intersections"][i]["virtual"]]
        self.id2intersection = dict()
        for inter in self.intersections:
            self.id2intersection[inter.id] = inter
        self.intersection_ids = [i.id for i in self.intersections]
        for inter in self.intersections:
            inter.observe()
        self._update_infos()

    def get_info(self, info):
        return self.info[info]

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()
    
    def get_vehicles(self):
        pass

    def get_lane_vehicle_count(self):
        # TODO: This is the test, try observe from full_observation later
        result = {k: 0 for k in self.all_lanes}
        result_update = self.eng.get_lane_vehicle_count()
        for k in result_update.keys():
            result.update({k: result_update[k]})
        return result

    def get_pressure(self):
        pass

    def get_lane_waiting_time_count(self):
        # this is the test
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: intsec.full_observation[lane]['lane_waiting_time_count']})
        return result

    def get_lane_waiting_vehicle_count(self):
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: intsec.full_observation[lane]['lane_waiting_count']})
        return result

    def get_cur_phase(self):
        result = list()
        for intsec in self.intersections:
            result.append(intsec.current_phase)
        return result

    def get_average_travel_time(self):
        # TODO: wrap it to see if needed
        return self.eng.get_average_travel_time()

    def get_lane_vehicles(self):
        # provided directly from engine, but not working at the few steps since 0 car lanes won't whow in the dictionary
        result = {k: 0 for k in self.all_lanes}
        result_update = self.eng.get_lane_vehicle_count()
        for k in result_update.keys():
            result.update({k: result_update[k]})
        return result

    def get_lane_queue_length(self):
        # TODO: currently not working 
        result = dict()
        for inter in self.intersections:
            for key in inter.full_observation.keys():
                result.update({key: inter.full_observation[key]['queue_length']})
        return result

    def get_lane_delay(self):
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        # set speed limit to 11.11 by default
        cur_lane_records = dict()
        for inter in self.intersections:
            for k, v in inter.vehicles_cur[lane].items():
                cur_lane_records.update({k: v})

        lane_delay = dict()
        for key in cur_lane_records.keys():
            # could be non in the few first steps
            vehicles = cur_lane_records[key].keys()
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            speed_limit = self.lane_maxSpeed[key]
            for vehicle in vehicles:
                speed = vehicle['speed']
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = speed_limit
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[key] = 1 - lane_avg_speed / speed_limit
        return lane_delay


if __name__ == "__main__":
    world = World(os.path.join(os.getcwd(), 'configs/openengine1x1.cfg'), 12)
    warm_up_time = 3600
    start_time = time.time()
    for i in range(2):
        world.reset()
        for step in range(warm_up_time):
            for intersection in world.intersections:
                world.eng.set_ttl_phase(intersection.id, (int(world.eng.get_current_time()) // 30) % 4 + 1)
            state = world.eng.get_lane_vehicles()
            print("t: {}, v: {}".format(step, world.eng.get_vehicle_count()))
            world.step([random.randint(0, 7)])
        end_time = time.time()
    print('Runtime: ', end_time - start_time)
