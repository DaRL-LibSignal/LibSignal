import os
import sys
from math import atan2, pi
import xml.etree.cElementTree as ET

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit('No SUMO in environment path')
from common.registry import Registry

import json
import re
import copy

import sumolib
import traci

# TODO: change it in the future. Accoring to _get_direction
DIRECTION_RANK = {'N': 0, 'W': 1, 'S': 2, 'E': 3}


# TODO: revert x and y
def _get_direction(road):
    x = road[1][0] - road[0][0]
    y = road[1][1] - road[0][1]
    tmp = atan2(x, y)
    return tmp if tmp >= 0 else (tmp + 2 * pi)


def create_yellows(phases, yellow_length):
    new_phases = copy.copy(phases)
    yellow_dict = {}    # current phase + next phase keyed to corresponding yellow phase index
    # Automatically create yellow phases, traci will report missing phases as it assumes execution by index order
    for i in range(0, len(phases)):
        for j in range(0, len(phases)):
            if i != j:
                need_yellow, yellow_str = False, ''
                for sig_idx in range(len(phases[i].state)):
                    if (phases[i].state[sig_idx] == 'G' or phases[i].state[sig_idx] == 'g') and (phases[j].state[sig_idx] == 'r' or phases[j].state[sig_idx] == 's'):
                        need_yellow = True
                        yellow_str += 'y'
                    else:
                        yellow_str += phases[i].state[sig_idx]
                if need_yellow:  # If a yellow is required
                    new_phases.append(traci.trafficlight.Phase(yellow_length, yellow_str))
                    yellow_dict[str(i) + '_' + str(j)] = len(new_phases) - 1  # The index of the yellow phase in SUMO
    return new_phases, yellow_dict


class Intersection(object):
    def __init__(self, id, world, phases):
        self.id = id
        self.world = world
        self.eng = self.world.eng
        self.lanes = []
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = []
        self.in_roads = []
        self.road_lane_mapping = {}

        # links and phase information of each intersection
        self.current_phase = 0
        self.virtual_phase = 0  # see yellow phase as the same phase after changing
        self.next_phase = 0
        self.current_phase_time = 0

        self.yellow_phase_time = min([i.duration for i in self.eng.trafficlight.getAllProgramLogics(self.id)[0].phases])
        self.map_name = world.map  # TODO: try to add it to Registry later

        self.lane_links = world.eng.trafficlight.getControlledLinks(self.id)
        for link in self.lane_links:
            link = link[0]
            if link[0][:-2] not in self.road_lane_mapping.keys():
                self.road_lane_mapping.update({link[0][:-2]: []})  # assume less than 9 lanes in each road
                self.road_lane_mapping[link[0][:-2]].append(link[0])
                self.roads.append(link[0][:-2])
                self.outs.append(False)
                road = self.eng.lane.getShape(link[0])
                self.directions.append(_get_direction(road))
            elif link[0][:-2] in self.road_lane_mapping.keys() and link[0] not in self.road_lane_mapping[link[0][:-2]]:
                self.road_lane_mapping[link[0][:-2]].append(link[0])
            if link[1][:-2] not in self.road_lane_mapping.keys():
                self.road_lane_mapping.update({link[1][:-2]: []})  # assume less than 9 lanes in each road
                self.road_lane_mapping[link[1][:-2]].append(link[1])
                self.roads.append(link[1][:-2])
                self.outs.append(True)
                road = self.eng.lane.getShape(link[1])
                self.directions.append(_get_direction(road))
            elif link[1][:-2] in self.road_lane_mapping.keys() and link[1] not in self.road_lane_mapping[link[1][:-2]]:
                self.road_lane_mapping[link[1][:-2]].append(link[1])

        self._sort_roads()
        for key in self.road_lane_mapping.keys():
            for lane in self.road_lane_mapping[key]:
                self.lanes.append(lane)

        self.green_phases = phases
        self.phases = [i for i in range(len(phases))]
        self.phase_available_startlanes = []
        self.startlanes = []
        self.phase_available_lanelinks = []
        for r, p in enumerate(self.green_phases):
            tmp_lanelinks = []
            tmp_startane = []
            for n, i in enumerate(p.state):
                if i == 'G' or i == 's':
                    links = self.world.eng.trafficlight.getControlledLinks(self.id)[n][0]
                    tmp_lanelinks.append([links[0], links[1]])
                    if links[0] not in tmp_startane:
                        tmp_startane.append(links[0])
                    if links[0] not in self.startlanes:
                        self.startlanes.append(links[0])
            self.phase_available_startlanes.append(tmp_startane)
            self.phase_available_lanelinks.append(tmp_lanelinks)

        self.full_phases, self.yellow_dict = create_yellows(self.green_phases, self.yellow_phase_time)
        programs = self.eng.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.full_phases
        self.eng.trafficlight.setProgramLogic(self.id, logic)

        # dictionary of remembered features
        self.waiting_times = dict()
        self.full_observation = None
        self.last_step_vehicles = None

        # TODO: check .signals .full_observation .last_stet_vehicles need to be set or not

    def _sort_roads(self):
        order = sorted(range(len(self.roads)),
                       key=lambda i: (self.directions[i],
                                      self.outs[i] if self.world.RIGHT else not self.outs[i]))
        # TODO: check order [2,1,0] if self.RIGHT
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]  # TODO: check if its 4

    def reset(self):
        self.current_phase_time = 0
        self.virtual_phase = 0
        self.next_phase = 0
        self.waiting_times = dict()
        self.full_observation = None
        self.last_step_vehicles = None
        self.current_phase = self.get_current_phase()
        # eng is set in world
        programs = self.eng.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.full_phases
        self.eng.trafficlight.setProgramLogic(self.id, logic)

    def get_current_phase(self):
        return self.eng.trafficlight.getPhase(self.id)

    # TODO: change cityflow phase generator into phase property
    def prep_phase(self, new_phase):
        if self.get_current_phase() == new_phase:
            self.next_phase = self.get_current_phase()
            self.eng.trafficlight.setPhase(self.id, self.next_phase)
            self.current_phase = self.get_current_phase()
        else:
            self.next_phase = new_phase
            # find yellow phase between cur and next phases
            y_key = str(self.get_current_phase()) + '_' + str(new_phase)
            if y_key in self.yellow_dict:
                y_id = self.yellow_dict[y_key]
                self.eng.trafficlight.setPhase(self.id, y_id)  # phase turns into yellow here
                self.current_phase = self.get_current_phase()

    def _change_phase(self, phase):
        self.eng.trafficlight.setPhase(self.id, phase)
        self.current_phase = self.get_current_phase()

    def pseudo_step(self, action):
        # TODO: check if change state, yellow phase must less than minimum of action time
        # test yellow finished first
        self.virtual_phase = action
        if self.current_phase_time == self.yellow_phase_time:
            self._change_phase(action)
        else:
            if action != self.get_current_phase() and self.current_phase_time > self.yellow_phase_time:
                self.current_phase_time = 0
            if self.current_phase_time == 0:
                self.prep_phase(action)
            elif self.current_phase_time < self.yellow_phase_time:
                self._change_phase(self.current_phase)
            else:
                self._change_phase(action)

        self.current_phase_time += 1

    def observe(self, step_length, distance):
        full_observation = dict()
        all_vehicles = set()
        for lane in self.lanes:
            vehicles = []
            lane_measures = {'lane_waiting_time_count': 0, 'lane_waiting_count': 0, 'lane_count': 0, 'queue_length': 0}
            lane_vehicles = self._get_vehicles(lane, distance)
            for v in lane_vehicles:
                all_vehicles.add(v)
                if v in self.waiting_times:
                    self.waiting_times[v] += step_length
                elif self.eng.vehicle.getWaitingTime(v) > 0:
                    self.waiting_times[v] = self.eng.vehicle.getWaitingTime(v)
                v_measures = dict()
                v_measures['name'] = v
                v_measures['wait'] = self.waiting_times[v] if v in self.waiting_times else 0
                #TODO: CHEC ITS RIGHT CALCULATION?
                lane_measures['queue_length'] = lane_measures['queue_length'] + 1
                v_measures['speed'] = self.eng.vehicle.getSpeed(v)
                v_measures['position'] = self.eng.vehicle.getLanePosition(v)
                vehicles.append(v_measures)
                if v_measures['wait'] > 0:
                    lane_measures['lane_waiting_time_count'] += v_measures['wait']
                    lane_measures['lane_waiting_count'] += 1
                lane_measures['lane_count'] += 1
            lane_measures['vehicles'] = vehicles
            full_observation[lane] = lane_measures
        """
        full_observation['num_vehicles'] = all_vehicles
        if self.last_step_vehicles is None:
            full_observation['arrivals'] = full_observation['num_vehicles']
            full_observation['departures'] = set()
        else:
            full_observation['arrivals'] = self.last_step_vehicles.difference(all_vehicles)
            departs = all_vehicles.difference(self.last_step_vehicles)
            full_observation['departures'] = departs
            # Clear departures from waiting times
            for vehicle in departs:
                if vehicle in self.waiting_times: self.waiting_times.pop(vehicle)
        self.last_step_vehicles = all_vehicles
        """
        self.full_observation = full_observation

    def _get_vehicles(self, lane, max_distance):
        # TODO: reduce complexity -> find all vehicles within max_distance and on this lane
        detectable = []
        for v in self.eng.lane.getLastStepVehicleIDs(lane):
            path = self.eng.vehicle.getNextTLS(v)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:
                    detectable.append(v)
        return detectable


@Registry.register_world('sumo')
class World(object):
    def __init__(self, sumo_config, placeholder=0):
        with open(sumo_config) as f:
            sumo_dict = json.load(f)
        if sumo_dict['gui'] == "True":
            sumo_cmd = [sumolib.checkBinary('sumo-gui')]
        else:
            sumo_cmd = [sumolib.checkBinary('sumo')]
        if not sumo_dict.get('combined_file'):
            sumo_cmd += ['-n', os.path.join(sumo_dict['dir'], sumo_dict['roadnetFile']),
                         '-r', os.path.join(sumo_dict['dir'], sumo_dict['flowFile']),
                         '--no-warnings', sumo_dict['no_warning']]
        else:
            sumo_cmd += ['-c', os.path.join(sumo_dict['dir'], sumo_dict['combined_file']),
                         '--no-warnings', sumo_dict['no_warning']]
        self.net = os.path.join(sumo_dict['dir'], sumo_dict['roadnetFile'])
        self.route = os.path.join(sumo_dict['dir'], sumo_dict['flowFile'])
        self.sumo_cmd = sumo_cmd
        self.warning = sumo_dict['no_warning']
        print("building world...")
        self.connection_name = sumo_dict['name']
        self.map = sumo_dict['roadnetFile'].split('/')[-1].split('.')[0]
        if not sumo_dict['name']:
            traci.start(sumo_cmd)
            self.eng = traci
        else:
            traci.start(sumo_cmd, label=sumo_dict['name'])
            self.eng = traci.getConnection(sumo_dict['name'])
        # TODO: roadnet not implemented but not necessary
        self.RIGHT = True  # TODO: currently set to be true
        self.interval = sumo_dict['interval']
        self.step_ratio = 1  # TODO: register in Registry later
        self.step_length = 1  # should be 1 in our setting
        self.max_distance = 200 # TODO: set in registry
        # get all intersections (dict here)
        self.intersection_ids = self.eng.trafficlight.getIDList()
        # prepare phase information for each intersections
        self.green_phases = self.generate_valid_phase()

        # creating all intersections
        self.id2intersection = dict()
        self.intersections = []
        for ts in self.eng.trafficlight.getIDList():
            self.id2intersection[ts] = Intersection(ts, self, self.green_phases[ts])  # this IntSec has different phases
            self.intersections.append(self.id2intersection[ts])
        # TODO: to see if its necessary to test .intersections or .observe here
        # TODO: to see if pass observation and its shape by generator
        self.all_roads = []
        self.all_lanes = []
        for itsec in self.intersections:
            for road in itsec.road_lane_mapping.keys():
                if itsec.road_lane_mapping[road] and road not in self.all_roads:
                    # append road name into all_roads if road exists
                    self.all_roads.append(road)
                    for lane in itsec.road_lane_mapping[road]:
                        if lane not in self.all_lanes:
                            self.all_lanes.append(lane)

        # restart eng
        self.run = 0
        self.inside_vehicles = dict()
        # self.inside_vehicles_planned = self.get_plan_depart_time() # all vehicles appeared in rou.xml
        self.vehicles = dict()
        # self.vehicles_planned = dict()
        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)

        if not self.connection_name: traci.switch(self.connection_name)  # TODO: make sure what's this step doing
        traci.close()
        # self.connection_name = self.map + '-' + self.connection_name
        if not os.path.exists(os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                           self.connection_name)):
            os.mkdir(os.path.join(Registry.mapping['logger_mapping']['output_path'].path, self.connection_name))

        print('Connection ID', self.connection_name)

        self.info_functions = {
            "vehicles": self.get_vehicles,
            "lane_count": self.get_lane_vehicle_count,
            "lane_waiting_count": self.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.get_lane_vehicles,
            "time": self.get_current_time,
            "vehicle_distance": None,
            "pressure": self.get_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "vehicle_trajectory": None,
            "history_vehicles": None,
            "phase": self.get_cur_phase,
            "throughput": self.get_cur_throughput,
            "average_travel_time": None
        }
        self.fns = []
        self.info = {}
        # test generate observation information

    def generate_valid_phase(self):
        valid_phases = dict()
        for i in range(0, 500):    # TODO grab info. directly from tllogic python interface
            for lightID in self.intersection_ids:
                current_phase = self.eng.trafficlight.getRedYellowGreenState(lightID)
                if not lightID in valid_phases:
                    valid_phases[lightID] = []
                has_phase = False
                for phase in valid_phases[lightID]:
                    if phase == current_phase:
                        has_phase = True
                if not has_phase:
                    valid_phases[lightID].append(current_phase)
            self.step_sim()
        for ts in valid_phases:
            green_phases = []
            for phase in valid_phases[ts]:    # Convert to SUMO phase type
                if 'y' not in phase:
                    # condition2: avoid regarding yellow phase as green phase, which will happened in cityflow_convert file.
                    # if (phase.count('r') + phase.count('s') != len(phase)) and (3*phase.count('G') != len(phase)):
                    if phase.count('r') + phase.count('s') != len(phase):
                        green_phases.append(self.eng.trafficlight.Phase(self.step_length, phase))
            valid_phases[ts] = green_phases
        return valid_phases

    def step_sim(self):
        # The monaco scenario expects .25s steps instead of 1s, account for that here.
        for _ in range(self.step_ratio):
            self.eng.simulationStep()

    def step(self, action=None):
        # TODO: support interval != 1
        if action is not None:
            for i, intersection in enumerate(self.intersections):
                intersection.pseudo_step(action[i])
            self.step_sim()
        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)
        # TODO: register vehicles here
        entering_v = self.eng.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.get_current_time()})
            # traci.vehicle.setLaneChangeMode(v,256)
        exiting_v = self.eng.simulation.getArrivedIDList()
        for v in exiting_v:
            self.vehicles.update({v: self.get_current_time() - self.inside_vehicles[v]})
            # self.vehicles_planned.update({v: self.get_current_time() - self.inside_vehicles_planned[v]})
        self._update_infos()
        self.run += 1

    def reset(self):
        if self.run != 0:
            # TODO: test why need switch in original code
            traci.close()
        self.run = 0
        self.vehicles = dict()
        # self.vehicles_planned = dict()
        self.inside_vehicles = dict()
        # TODO: check when to close traci
        traci.start(self.sumo_cmd, label=self.connection_name)
        # TODO: set trip info output
        self.eng = traci.getConnection(self.connection_name)
        self.id2intersection = dict()
        self.intersections = []
        for ts in self.eng.trafficlight.getIDList():
            self.id2intersection[ts] = Intersection(ts, self, self.green_phases[ts])  # this IntSec has different phases
            self.intersections.append(self.id2intersection[ts])

        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)
        self._update_infos()
        # TODO: check if its the problem
        entering_v = self.eng.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.get_current_time()})

    def get_current_time(self):
        result = self.eng.simulation.getTime()
        return result

    def get_vehicles(self):
        # assert len(self.vehicles) == len(self.vehicles_planned)
        result = 0
        result_planned = 0
        count = 0
        for v in self.vehicles.keys():
            count += 1
            result += self.vehicles[v]
            result_planned += self.vehicles[v]
            # result_planned += self.vehicles_planned[v]
        if count == 0:
            return [0, 0]
        else:
            return [result/count, result_planned/count]

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if fn not in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception(f'Info function {fn} not implemented')

    def get_info(self, info):
        return self.info[info]

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_lane_vehicle_count(self):
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: intsec.full_observation[lane]['lane_count']})
        return result

    def get_pressure(self):
        pressures = dict()
        lane_vehicles = self.get_lane_vehicle_count()
        for i in self.intersections:
            pressure = 0
            for road in i.in_roads:
                for k in i.road_lane_mapping[road]:
                    pressure += lane_vehicles[k]
            for road in i.out_roads:
                for k in i.road_lane_mapping[road]:
                    pressure -= lane_vehicles[k]
            pressures[i.id] = pressure
        return pressures

    def get_lane_waiting_time_count(self):
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
        result = []
        for intsec in self.intersections:
            result.append(intsec.get_current_phase())
        return result

    def get_average_travel_time(self):
        """
        return: [real travel time, planned travel time(aligned with Cityflow)]
        """
        return self.get_vehicles()

    def get_lane_vehicles(self):
        result = dict()
        for inter in self.intersections:
            for key in inter.full_observation.keys():
                result.update({key: inter.full_observation[key]})
        return result

    def get_lane_queue_length(self):
        #TODO: CHECK DEFINATION
        result = dict()
        for inter in self.intersections:
            for key in inter.full_observation.keys():
                result.update({key: inter.full_observation[key]['queue_length']})
        return result

    def get_lane_delay(self):
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        # set speed limit to 11.11 by default
        lane_vehicles = self.get_lane_vehicles()
        lane_delay = dict()
        for key in lane_vehicles.keys():
            vehicles = lane_vehicles[key]['vehicles']
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            speed_limit = self.eng.lane.getMaxSpeed(key)
            for vehicle in vehicles:
                speed = vehicle['speed']
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = speed_limit
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[key] = 1 - lane_avg_speed / speed_limit
        return lane_delay

    # def get_plan_depart_time(self):
    #     """
    #     Get planned depart time for all vehicles appeared in sumo.rou.xml file.
    #     In SUMO and Cityflow, travel time = arriving time-planned depart time.
    #     Note: Not real depart time, but planned depart time.
    #     return: planned depart time of all vehicles.
    #     """
    #     vehicles_all = dict()
    #     tree = ET.parse(self.route)
    #     root = tree.getroot()
    #     vehicles_all.update({obj.attrib['id']: int(float(obj.attrib['depart'])) \
    #         for obj in root.iter('vehicle')})
    #     return vehicles_all

    def get_cur_throughput(self):
        throughput = len(self.vehicles)
        # TODO: check if only trach left cars
        return throughput




