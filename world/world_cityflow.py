import json
import os
import cityflow
from common.registry import Registry

import numpy as np
from math import atan2, pi
import math

class Intersection(object):
    '''
    Intersection Class is mainly used for describing crossing information and defining acting methods.
    '''
    def __init__(self, intersection, world):
        self.id = intersection["id"]
        self.world = world
        self.eng = self.world.eng


        # incoming and outgoing roads of each intersection, clock-wise order from North
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = None
        self.in_roads = None

        # map_name = Registry.mapping['world_mapping']['setting'].param['network']
        # self.lane_order_cf = None
        # self.lane_order_sumo = None
        # if 'signal_config' in Registry.mapping['world_mapping']['setting'].param.keys():
        #     if 'N' in Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'].keys():
        #         self.lane_order_cf = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order']
        #         self.lane_order_sumo = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['sumo_order']
        #     else:
        #         self.lane_order_cf = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'][self.id]
        #         self.lane_order_sumo = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['sumo_order'][self.id]

        # links and phase information of each intersection
        self.current_phase = 0
        self.roadlinks = []
        self.lanelinks_of_roadlink = []
        self.startlanes = []
        self.lanelinks = []
        self.phase_available_roadlinks = []
        self.phase_available_lanelinks = []
        self.phase_available_startlanes = []

        self.if_sumo = True if "gt_virtual" in intersection else False

        # create yellow phases
        # in cityflow, yellow phases' id is 0
        # in sumo, yellow phases' id is the first appeared in phases 
        phases = intersection["trafficLight"]["lightphases"]
        self.all_phases = [i for i in range(len(phases))]
        if self.if_sumo:
            self.yellow_phase_time = min([i['time'] for i in phases])
            self.yellow_phase_id = [i for i in range(len(phases)) if phases[i]['time']==self.yellow_phase_time]
            self.phases = [i for i in range(len(phases)) if phases[i]['time']!=self.yellow_phase_time]
        else:
            self.yellow_phase_id = [0]
            self.yellow_phase_time = 5
            self.phases = [i for i in range(len(phases)) if not i in self.yellow_phase_id] # mapping from model output to cityflow phase id
        # parsing links and phases
        for roadlink in intersection["roadLinks"]:
            self.roadlinks.append((roadlink["startRoad"], roadlink["endRoad"]))
            lanelinks = []
            for lanelink in roadlink["laneLinks"]:
                startlane = roadlink["startRoad"] + "_" + str(lanelink["startLaneIndex"])
                self.startlanes.append(startlane)
                endlane = roadlink["endRoad"] + "_" + str(lanelink["endLaneIndex"])
                lanelinks.append((startlane, endlane))
            self.lanelinks.extend(lanelinks)
            self.lanelinks_of_roadlink.append(lanelinks)

        self.startlanes = list(set(self.startlanes))

        for i in self.phases:
            phase = phases[i]
            self.phase_available_roadlinks.append(phase["availableRoadLinks"])
            phase_available_lanelinks = []
            phase_available_startlanes = []
            for roadlink_id in phase["availableRoadLinks"]:
                lanelinks_of_roadlink = self.lanelinks_of_roadlink[roadlink_id]
                phase_available_lanelinks.extend(lanelinks_of_roadlink)
                for lanelinks in lanelinks_of_roadlink:
                    phase_available_startlanes.append(lanelinks[0])
            self.phase_available_lanelinks.append(phase_available_lanelinks)
            phase_available_startlanes = list(set(phase_available_startlanes))
            self.phase_available_startlanes.append(phase_available_startlanes)

        self.reset()

    def insert_road(self, road, out):
        '''
        insert_road
        It's used to append a road into self.road and add the corresponding direction with the added road.
        
        :param road: newly added road
        :param out: newly added out
        :return: None
        '''
        self.roads.append(road)
        self.outs.append(out)
        self.directions.append(self._get_direction(road, out))

    def sort_roads(self):
        '''
        sort_roads
        Sort roads information by arranging an order.
        
        :return: None
        '''
        # self.world.RIGHT: decide whether to sort from right side, 
        # currently always set to true due to CityFlow's mechanism.
        order = sorted(range(len(self.roads)),
                       key=lambda i: (self.directions[i], self.outs[i] if self.world.RIGHT else not self.outs[i]))
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]

    def _change_phase(self, phase, interval, typ='init'):
        '''
        _change_phase
        Change current phase and calculate time duration of current phase.
        
        :param phase: true phase id (including yellows)
        :param interval: the non-acting time slice
        :param typ: calculation type of current phase time, 
        'init' means calculate from scratch, 
        'add' means current phase time add interval time.
        :return: None
        '''
        self.eng.set_tl_phase(self.id, phase)
        self._current_phase = phase
        if typ == 'add':
            self.current_phase_time += interval
        else:
            self.current_phase_time = interval

    def step(self, action, interval):
        '''
        step
        Take relative actions according to interval.
        
        :param action: the changes to take
        :param interval: the non-acting time slice
        :return: None
        '''
        # if current phase is yellow, then continue to finish the yellow phase
        # recall self._current_phase means true phase id (including yellows)
        # self.current_phase means phase id in self.phases (excluding yellow)
        if self._current_phase in self.yellow_phase_id:
            if self.current_phase_time == self.yellow_phase_time:
                self._change_phase(self.phases[self.action_before_yellow], interval,'add')
                # if self.if_sumo:
                #     self._change_phase(self.phases[self.action_before_yellow], interval,'add')
                # else:
                #     self._change_phase(self.phases[self.action_before_yellow], interval)
                self.current_phase = self.action_before_yellow
                self.action_executed = self.action_before_yellow
            else:
                self.current_phase_time += interval
        else:
            if action == self.current_phase:
                self.current_phase_time += interval
            else:
                if self.yellow_phase_time > 0:
                    # yellow(red) phase is arranged behind each green light
                    if self.if_sumo:
                        assert (self._current_phase+1)%len(self.all_phases) in self.yellow_phase_id
                        self._change_phase((self._current_phase+1)%len(self.all_phases), interval)
                    else:
                        self._change_phase(self.yellow_phase_id[0], interval)
                    self.action_before_yellow = action
                else:
                    self._change_phase(self.phases[action], interval)
                    self.current_phase = action
                    self.action_executed = action

    def reset(self):
        '''
        reset
        Reset information, including current_phase, action_before_yellow and action_executed, etc.

        :param: None
        :return: None
        '''
        # record phase info
        self.current_phase = 0  # phase id in self.phases (excluding yellow)
        if len(self.phases) == 0:
            self._current_phase = 0
        else:
            self._current_phase = self.phases[0]  # true phase id (including yellow)
        self.eng.set_tl_phase(self.id, self._current_phase)
        self.current_phase_time = 0
        self.action_before_yellow = None
        self.action_executed = None

    # TODO: THIS IS Y/X  But we keep it right now
    def _get_direction(self, road, out=True):
        if out:
            x = road["points"][1]["x"] - road["points"][0]["x"]
            y = road["points"][1]["y"] - road["points"][0]["y"]
        else:
            x = road["points"][-2]["x"] - road["points"][-1]["x"]
            y = road["points"][-2]["y"] - road["points"][-1]["y"]
        tmp = atan2(x, y)
        return tmp if tmp >= 0 else (tmp + 2 * pi)


@Registry.register_world('cityflow')
class World(object):
    '''
    World Class is mainly used for creating a CityFlow engine and maintain information about CityFlow world.
    '''

    def __init__(self, cityflow_config, thread_num, **kwargs):
        print("building world...")
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        self.roadnet = self._get_roadnet(cityflow_config)
        self.RIGHT = True  # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism
        self.interval = cityflow_config["interval"]

        # get all non virtual intersections
        # judge whether the file is convert from sumo file,
        # (in sumo_convert file, the "virtual" value of all intersections are set to "False"),
        # if so, then must use "gt_virtual" to create non-virtual intersections,
        # if not, just use "virtual" to create.
        if_sumo = True if "gt_virtual" in self.roadnet["intersections"][0] else False
        if_cf_virtual = True if "cf_gt_virtual" in self.roadnet["intersections"][0] else False
        if if_sumo:
            self.intersections = [i for i in self.roadnet["intersections"] if not i["gt_virtual"]]
        else:
            if if_cf_virtual:
                self.intersections = [i for i in self.roadnet["intersections"] if not i["cf_gt_virtual"]]
            else:
                self.intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersection_ids = [i["id"] for i in self.intersections]

        # create non-virtual Intersections
        print("creating intersections...")
        if if_sumo:
            non_virtual_intersections = [i for i in self.roadnet["intersections"] if not i["gt_virtual"]]
        else:
            if if_cf_virtual:
                non_virtual_intersections = [i for i in self.roadnet["intersections"] if not i["cf_gt_virtual"]]
            else:
                non_virtual_intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersections = [Intersection(i, self) for i in non_virtual_intersections]
        # if len(self.intersections) == 6:
        #     self.intersections = self.intersections[0:5]
        self.intersection_ids = [i["id"] for i in non_virtual_intersections]
        # if len(self.intersection_ids) == 6:
        #     self.intersection_ids = self.intersection_ids[0:5]
        self.id2intersection = {i.id: i for i in self.intersections}
        self.id2idx = {i: idx for idx,i in enumerate(self.id2intersection)}
        print("intersections created.")

        # id of all roads and lanes
        print("parsing roads...")
        self.all_roads = []
        self.all_lanes = []
        self.all_lanes_speed = {}
        self.lane_length = {}


        for road in self.roadnet["roads"]:
            self.all_roads.append(road["id"])
            i = 0
            road_l = self.get_road_length(road)
            for lane in road["lanes"]:
                self.all_lanes.append(road["id"] + "_" + str(i))
                self.all_lanes_speed[road["id"] + "_" + str(i)] = lane['maxSpeed']
                self.lane_length[road["id"] + "_" + str(i)] = road_l
                i += 1

            iid = road["startIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, True)
            iid = road["endIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, False)

        for i in self.intersections:
            i.sort_roads()

        print("roads parsed.")

        # initializing info functions
        self.info_functions = {
            "vehicles": (lambda: self.eng.get_vehicles(include_waiting=True)),
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "time": self.eng.get_current_time,
            "vehicle_distance": self.eng.get_vehicle_distance,
            "pressure": self.get_pressure,
            "lane_pressure": self.get_lane_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "real_delay": self.get_real_delay,
            "vehicle_trajectory": self.get_vehicle_trajectory,
            "history_vehicles": self.get_history_vehicles,
            "phase": self.get_cur_phase,
            "throughput": self.get_cur_throughput,
            "averate_travel_time": self.get_average_travel_time
            # "action_executed": self.get_executed_action
        }
        self.fns = []
        self.info = {}
        self.vehicle_waiting_time = {}  # key: vehicle_id, value: the waiting time of this vehicle since last halt.
        self.vehicle_trajectory = {}  # key: vehicle_id, value: [[lane_id_1, enter_time, time_spent_on_lane_1], ... , [lane_id_n, enter_time, time_spent_on_lane_n]]
        self.history_vehicles = set()
        self.real_delay= {}

        # # get in_lanes and out_lanes
        self.in_lanes, self.out_lanes = self.get_in_out_lanes()

        # record lanes' vehicles to calculate arrive_leave_time
        self.dic_lane_vehicle_previous_step = {key: None for key in self.all_lanes}
        self.dic_lane_vehicle_current_step = {key: None for key in self.all_lanes}
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        print("world built.")

    def reset_vehicle_info(self):
        '''
        reset_vehicle_info
        Reset vehicle infos, including waiting_time, trajectory, etc.

        :param: None
        :return: None
        '''
        self.vehicle_waiting_time = {}  # key: vehicle_id, value: the waiting time of this vehicle since last halt.
        self.vehicle_trajectory = {}  # key: vehicle_id, value: [[lane_id_1, enter_time, time_spent_on_lane_1], ... , [lane_id_n, enter_time, time_spent_on_lane_n]]
        self.history_vehicles = set()
        self.real_delay= {}
        self.dic_lane_vehicle_previous_step = {key: None for key in self.all_lanes}
        self.dic_lane_vehicle_current_step = {key: None for key in self.all_lanes}
        self.dic_vehicle_arrive_leave_time = dict()

    def _update_arrive_time(self, list_vehicle_arrive):
        '''
        _update_arrive_time
        Update enter time of vehicles.

        :param list_vehicle_arrive: vehicles' id that have entered in roadnet
        :return: None
        '''
        ts = self.eng.get_current_time()
        # init vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan,
                                                               "cost_time": np.nan}
            else:
                # print("vehicle: %s already exists in entering lane!"%vehicle)
                pass

    def _update_left_time(self, list_vehicle_left):
        '''
        _update_left_time
        Update left time of vehicles.

        :param list_vehicle_left: vehicles' id that have left from roadnet
        :return: None
        '''
        ts = self.eng.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
                self.dic_vehicle_arrive_leave_time[vehicle]["cost_time"] = ts - \
                                                                           self.dic_vehicle_arrive_leave_time[vehicle][
                                                                               "enter_time"]
            except KeyError:
                print("vehicle not recorded when entering!")

    def update_current_measurements(self):
        '''
        update_current_measurements
        Update information, including enter time of vehicle, left time of vehicle, lane id that vehicles are running, etc.
        
        :param: None
        :return: None
        '''
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                if value:
                    list_lane_vehicle.extend(value)
            return list_lane_vehicle

        # contain outflow lanes
        self.dic_lane_vehicle_current_step = self.eng.get_lane_vehicles()

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)
        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(
            set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
    

    def get_cur_throughput(self):
        '''
        get_cur_throughput
        Get vehicles' count in the whole roadnet at current step.

        :param: None
        :return throughput: throughput in the whole roadnet at current step
        '''
        throughput = 0
        for dic in self.dic_vehicle_arrive_leave_time:
            vehicle = self.dic_vehicle_arrive_leave_time[dic]
            if (not np.isnan(vehicle["cost_time"])) and vehicle["leave_time"] <= self.eng.get_current_time():
                throughput += 1
        return throughput

    def get_executed_action(self):
        '''
        get_executed_action
        Get executed action of each intersection at current step.

        :param: None
        :return actions: executed action of each intersection at current step
        '''
        actions = []
        for i in self.intersections:
            actions.append(i.action_executed)
        return actions

    def get_cur_phase(self):
        '''
        get_cur_phase
        Get current phase of each intersection.

        :param: None
        :return phases: current phase of each intersection
        '''
        phases = []
        for i in self.intersections:
            phases.append(i.current_phase)
        return phases

    def get_pressure(self):
        '''
        get_pressure
        Get pressure of each intersection. 
        Pressure of an intersection equals to number of vehicles that in in_lanes minus number of vehicles that in out_lanes.
        
        :param: None
        :return pressures: pressure of each intersection
        '''
        vehicles = self.eng.get_lane_vehicle_count()
        pressures = {}
        for i in self.intersections:
            pressure = 0
            in_lanes = []
            for road in i.in_roads:
                from_zero = (road["startIntersection"] == i.id) if self.RIGHT else (
                        road["endIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))
            out_lanes = []
            for road in i.out_roads:
                from_zero = (road["endIntersection"] == i.id) if self.RIGHT else (
                        road["startIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    out_lanes.append(road["id"] + "_" + str(n))
            for lane in vehicles.keys():
                if lane in in_lanes:
                    pressure += vehicles[lane]
                if lane in out_lanes:
                    pressure -= vehicles[lane]
            pressures[i.id] = pressure
        return pressures
    
    def get_in_out_lanes(self):
        in_lanes = []
        out_lanes = []
        for i in self.intersections:
            for road in i.in_roads:
                from_zero = (road["startIntersection"] == i.id) if self.RIGHT else (
                        road["endIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))
            for road in i.out_roads:
                from_zero = (road["endIntersection"] == i.id) if self.RIGHT else (
                        road["startIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    out_lanes.append(road["id"] + "_" + str(n))
        # add in_lanes of virtual intersections which can be regarded as out_lanes of non-virtual intersections.
        for lane in self.all_lanes:
            if lane not in out_lanes:
                out_lanes.append(lane)
        return in_lanes, out_lanes

    def get_lane_pressure(self):
        '''
        get_lane_pressure
        Get pressure of each lane in an intersection. 
        Pressure of each lane equals to number of vehicles that in the in_lane minus number of vehicles that in out_lane.
        
        :param: None
        :return pressures: pressure of each lane
        '''
        lvc = self.eng.get_lane_vehicle_count()
        pressures = {}
        pressures = {x:0 for x in self.in_lanes}
        for inter_obj in self.intersections:
            pressure = []
            for start, end in inter_obj.lanelinks:
                pressures[start] += lvc[start]
                pressures[start] -= lvc[end]
        return pressures

    def get_vehicle_lane(self):
        '''
        get_vehicle_lane
        Get current lane id of each vehicle that is running.

        :param: None
        :return vehicle_lane: current lane id of each vehicle that is running
        '''
        # get the current lane of each vehicle. {vehicle_id: lane_id}
        vehicle_lane = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        for lane in self.all_lanes:
            for vehicle in lane_vehicles[lane]:
                vehicle_lane[vehicle] = lane
        return vehicle_lane

    def get_vehicle_waiting_time(self):
        '''
        get_vehicle_waiting_time
        Get waiting time of vehicles according to vehicle's speed. 
        If a vehicle's speed less than 0.1m/s, then its waiting time would be added 1s.
        
        :param: None
        :return vehicle_waiting_time: waiting time of vehicles
        '''
        # the waiting time of vehicle since last halt.
        vehicles = self.eng.get_vehicles(include_waiting=False)
        vehicle_speed = self.eng.get_vehicle_speed()
        for vehicle in vehicles:
            if vehicle not in self.vehicle_waiting_time.keys():
                self.vehicle_waiting_time[vehicle] = 0
            if vehicle_speed[vehicle] < 0.1:
                self.vehicle_waiting_time[vehicle] += 1
            else:
                self.vehicle_waiting_time[vehicle] = 0
        return self.vehicle_waiting_time

    def get_lane_waiting_time_count(self):
        '''
        get_lane_waiting_time_count
        Get waiting time of vehicles in each lane.
        
        :param: None
        :return lane_waiting_time: waiting time of vehicles in each lane
        '''
        # the sum of waiting times of vehicles on the lane since their last halt.
        lane_waiting_time = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        vehicle_waiting_time = self.get_vehicle_waiting_time()
        for lane in self.all_lanes:
            lane_waiting_time[lane] = 0
            for vehicle in lane_vehicles[lane]:
                lane_waiting_time[lane] += vehicle_waiting_time[vehicle]
        return lane_waiting_time

    def get_lane_delay(self):
        '''
        get_lane_delay
        Get approximate delay of each lane. 
        Approximate delay of each lane equals to (1 - lane_avg_speed)/lane_speed_limit.
        
        :param: None
        :return lane_delay: approximate delay of each lane
        '''
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_delay = {}
        lanes = self.all_lanes
        vehicle_speed = self.eng.get_vehicle_speed()

        for lane in lanes:
            vehicles = lane_vehicles[lane]
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            for vehicle in vehicles:
                speed = vehicle_speed[vehicle]
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = self.all_lanes_speed[lane]
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[lane] = 1 - lane_avg_speed / self.all_lanes_speed[lane]
        return lane_delay

    def get_vehicle_trajectory(self):
        '''
        get_vehicle_trajectory
        Get trajectory of vehicles that have entered in roadnet, including vehicle_id, enter time, leave time or current time.
        
        :param: None
        :return vehicle_trajectory: trajectory of vehicles that have entered in roadnet
        '''
        # lane_id and time spent on the corresponding lane that each vehicle went through
        vehicle_lane = self.get_vehicle_lane() # get vehicles on tne roads except turning
        vehicles = self.eng.get_vehicles(include_waiting=False)
        for vehicle in vehicles:
            if vehicle not in self.vehicle_trajectory:
                self.vehicle_trajectory[vehicle] = [[vehicle_lane[vehicle], int(self.eng.get_current_time()), 0]]
            else:
                if vehicle not in vehicle_lane.keys(): # vehicle is turning
                    continue
                if vehicle_lane[vehicle] == self.vehicle_trajectory[vehicle][-1][0]: # vehicle is running on the same lane 
                    self.vehicle_trajectory[vehicle][-1][2] += 1
                else: # vehicle has changed the lane
                    self.vehicle_trajectory[vehicle].append(
                        [vehicle_lane[vehicle], int(self.eng.get_current_time()), 0])
        return self.vehicle_trajectory

    def get_history_vehicles(self):
        '''
        get_history_vehicles
        Get vehicles that have entered in roadnet.
        
        :param: None
        :return history_vehicles: information of vehicles that have entered in roadnet.
        '''
        self.history_vehicles.update(self.eng.get_vehicles())
        return self.history_vehicles

    def _get_roadnet(self, cityflow_config):
        '''
        _get_roadnet
        Read information from roadnet file in the config file.
        
        :param cityflow_config: config file of a roadnet
        :return roadnet: information of a roadnet
        '''

        """
        read information from roadnet file in the config file
        generate roadnet dictionary based on provide configuration file
        functions borrowed form openengine CBEngine.py
        Details:
        collect roadnet information.
        {1-'intersections'-(len=N_intersections):
            {11-'id': name of the intersection,
             12-'point': 121: {'x', 'y'}(intersection at this position),
             13-'width': intersection width,
             14-'roads'(len=N_roads controlled by this intersection): name of road
             15-'roadLinks'(len=N_road links): 
                {151-'type': direction type(go_straight, turn_left, turn_right, turn_U),  # TODO: check turn_u
                 152-'startRoad': start road name,
                 153-'endRoad': end road name,
                 154-'direction': int(same as type)
                 155-'laneLinks(len-N_lane links of road): 
                    {1551-'startLaneIndex': int(lane index in start road),
                     1552-'endLaneIndex': int(lane index in end road),
                     1553-'points(N_points alone this lane': {'x', 'y'}(point pos)
                     }
                 },
             16-'trafficLight: 
                {161-'roadLinkIndices'(len=N_road links): [],
                 162-'light phases'(len=N_phases): {1621-'time': int(time long),
                                                    1622-'availableRoadLinks'(len=N_working_roads): []
                                                    }
                 },
             17-'virtual': bool
             },
         2-'roads'-(len=N_roads ): 
            {21-'id': name of the road,
             22-'points': [221: {'x', 'y'}(start pos), 222: {'x', 'y'}(end pos)],
             23-'lanes'-(N_lanes in this road): 
                231: [{'width': lane width, 'maxSpeed': max speed of each car on this lane}]
                 232-'startIntersection': lane start,
                 233-'endIntersection': lane end
                 }
             }
         }
        """
        #roadnet_file = osp.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        roadnet_file = os.path.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

    def subscribe(self, fns):
        '''
        subscribe
        Subscribe information you want to get when training the model.
        
        :param fns: information name you want to get
        :return: None
        '''
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)

    def step(self, actions=None):
        '''
        step
        Take relative actions and update information, 
        including global information, measurements and trajectory, etc.
        
        :param actions: actions list to be executed at all intersections at the next step
        :return: None
        '''
        #  update previous measurement
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step

        if actions is not None:
            for i, action in enumerate(actions):
                self.intersections[i].step(action, self.interval)
        self.eng.next_step()
        self._update_infos()
        # update current measurement
        self.update_current_measurements()
        self.vehicle_trajectory = self.get_vehicle_trajectory()

    def reset(self):
        '''
        reset
        reset information, including waiting_time, trajectory, etc.
       
        :param: None
        :return: None
        '''
        self.eng.reset()
        for I in self.intersections:
            I.reset()
        self._update_infos()
        # reset vehicles info
        self.reset_vehicle_info()

    def _update_infos(self):
        '''
        _update_infos
        Update global information after reset or each step.
        
        :param: None
        :return: None
        '''
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        '''
        get_info
        Get specific information.
        
        :param info: the name of the specific information
        :return _info: specific information
        '''
        _info = self.info[info]
        return _info

    def get_average_travel_time(self):
        '''
        get_average_travel_time
        Get average travel time of all vehicles.
        
        :param: None
        :return tvg_time: average travel time of all vehicles
        '''
        tvg_time = self.eng.get_average_travel_time()
        return tvg_time

    def get_lane_queue_length(self):
        '''
        get_lane_queue_length
        Get queue length of all lanes in the traffic network.
        
        :param: None
        :return lane_q_length: queue length of all lanes
        '''
        lane_q_length = self.eng.get_lane_waiting_vehicle_count()
        return lane_q_length

    def get_road_length(self, road):
        '''
        get_road_length
        Calculate the length of a road.
        
        :param road: information about a road
        :return road_length: length of a specific road
        '''
        point_x = road['points'][0]['x'] - road['points'][1]['x']
        point_y = road['points'][0]['y'] - road['points'][1]['y']
        road_length = math.sqrt((point_x**2)+(point_y**2))
        return road_length
    
    def get_real_delay(self):
        '''
        get_real_delay
        Calculate average real delay. 
        Real delay of a vehicle is defined as the time a vehicle has traveled within the environment minus the expected travel time.
        
        :param: None
        :return avg_delay: average real delay of all vehicles
        '''
        self.vehicle_trajectory = self.get_vehicle_trajectory()
        for v in self.vehicle_trajectory:
            # get road level routes of vehicle
            routes = self.vehicle_trajectory[v] # lane_level
            for idx,lane in enumerate(routes):
                # speed = min(self.all_lanes_speed[lane[0]], float(info['speed']))
                speed = min(self.all_lanes_speed[lane[0]], 11.11)
                lane_length = self.lane_length[lane[0]]
                if idx == len(routes)-1: # the last lane
                    # judge whether the vehicle run over the whole lane.
                    dis = self.eng.get_vehicle_distance()
                    lane_length = dis[v] if v in dis.keys() else lane_length
                planned_tt = float(lane_length)/speed
                real_delay = lane[-1] - planned_tt if lane[-1]>planned_tt else 0.
                if v not in self.real_delay.keys():
                    self.real_delay[v] = real_delay
                else:
                    self.real_delay[v] += real_delay

        avg_delay = 0.
        count = 0
        for dic in self.real_delay.items():
            avg_delay += dic[1]
            count += 1
        avg_delay = avg_delay / count
        return avg_delay
        

if __name__ == "__main__":
    world = World("/mnt/d/Cityflow/tools/generator/configs.json", thread_num=1)
    # print(len(world.intersections[0].startlanes))
    print(world.intersections[0].phase_available_startlanes)