import numpy as np
import math
# from .base import BaseGenerator


class IntersectionVehicleGenerator():
    '''
    Generate state or reward based on statistics of intersection vehicles.

    :param world: World object
    :param I: Intersection object
    :param fns: list of statistics to get,"vehicle_trajectory", "lane_vehicles", "history_vehicles" is needed for result "passed_count" and "passed_time_count", 
                                    "vehicle_distance", "lane_vehicles" is needed for result "vehicle_map", 
                                    "phase" is needed for result "cur_phase". 
    :param targets: list of results to return, currently support "vehicle_map": map of vehicles: an image representation of vehicles' position in this intersection. 
                                                           "passed_count": total number of vehicles that passed the intersection during time interval ∆t after the last action. 
                                                           "passed_time_count": total time (in minutes) spent on approaching lanes of vehicles that passed the intersection during time interval ∆t after the last action. 
                                                           "cur_phase": current phase of the intersection (not before yellow phase). 
             See section 4.2 of the intelliLight paper[Hua Wei et al, KDD'18] for more detailed description on these targets.
    :param negative: boolean, whether return negative values (mostly for Reward).
    :param time_interval: use to calculate
    '''
    def __init__(self, world, I, fns=("vehicle_trajectory", "lane_vehicles", "history_vehicles", "vehicle_distance"), targets=("vehicle_map"), negative=False):
        self.world = world
        self.I = I

        # get cur phase of the intersection
        self.phase = I.current_phase

        # get lanes of intersections
        self.lanes = []
        self.in_lanes = []
        self.road_starting_points = {}
        roads = I.roads
        for road in roads:
            from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
            self.road_starting_points[road["id"]] = road["points"][0]
            self.lanes.append([ road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
            if from_zero:
                self.in_lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])

        self.all_lanes = [n for a in self.lanes for n in a ]
        self.all_in_lanes = [n for a in self.in_lanes for n in a]

        # print(self.all_lanes)
        # print(self.all_in_lanes)

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns
        self.targets = targets

        self.result_functions = {
            "passed_count": self.passed_count,
            "passed_time_count": self.passed_time_count,
            "vehicle_map": self.vehicle_map,
            "cur_phase": self.cur_phase
        }

        self.negative = negative

    def if_vehicle_passed_intersection(self, vehicle_trajectory, current_time, action_interval):
        '''
        if_vehicle_passed_intersection
        Judge whether a vehicle passes through intersection during time interval ∆t after the last action.
        
        :param vehicle_trajectory: trajectory of a vehicle
        :param current_time: current time of simulation
        :param action_interval: time duration between change actions
        :return result: boolean, whether the vehicle passes through intersection
        '''
        def get_target_trajectory(trajectory, last_time, current_time):
            target = []

            if len(trajectory) == 0:
                return target

            if trajectory[-1][1] + trajectory[-1][2] < current_time and trajectory[-1][1] + trajectory[-1][2] > last_time: # means this vehicle left the simulator or currently in an intersection in this period
                target.append(trajectory[-1])
                target.append(['left'])
                return target

            for i, traj in enumerate(trajectory):
                if traj[1] > last_time:
                    target.append(traj)
                    if traj[1] - traj[2] > last_time and i > 0:
                        target.append(trajectory[i - 1])

            return target
        last_time = current_time - action_interval

        current_trajectory = get_target_trajectory(vehicle_trajectory, last_time, current_time)
        for i in range(len(current_trajectory)):
            if current_trajectory[i][0] in self.all_in_lanes:
                if i + 1 < len(current_trajectory):
                    return 1
        return 0


    def get_passed_vehicles(self, fns):
        '''
        get_passed_vehicles
        Get the total number of vehicles that pass through intersections.
        
        :param fns: information of current intersection, including vehicle_trajectory, lane_vehicles, etc
        :return passed_vehicles: the number of vehicles that pass through intersections.
        '''
        vehicle_trajectory = fns["vehicle_trajectory"]
        history_vehicles = fns["history_vehicles"]

        passed_vehicles = []
        for vehicle in history_vehicles:
            if self.if_vehicle_passed_intersection(vehicle_trajectory[vehicle], current_time=self.time, action_interval=self.action_interval):
                passed_vehicles.append(vehicle)
        return passed_vehicles


    def get_vehicle_position(self, distance, lane):
        '''
        get_vehicle_position
        Get the location of vehicles in the roadnet.
        
        :param distance: calculate position of vehicles within the limited distance of a lane
        :param lane: lane id
        :return result: the location of vehicles appearing in the road network
        '''
        road = lane[:-2]
        # start_point = list(self.world.all_roads_starting_point[road].values())
        start_point = list(self.road_starting_points[road].values())
        # 0 right, 1 up, 2 left, 3 down
        direction_code = int(road[-1])
        # 0 x-axis, 1 y-axis
        way = direction_code % 2
        # 0 1 -> 1,    2 3 -> -1
        direction = -((direction_code * 2 - 3) / abs(direction_code * 2 - 3))
        cur_position = start_point.copy()
        distance = int(distance)
        cur_position[way] += direction * distance
        result = tuple(cur_position)
        return result


    def passed_count(self, fns):
        '''
        passed_count
        Get the total number of vehicles that passed the intersection during time interval ∆t after the last action.
        
        :param fns: information of current intersection, including vehicle_trajectory, lane_vehicles, etc
        :return result: the total number of vehicles that passed the intersection
        '''
        passed_vehicles = self.get_passed_vehicles(fns)
        result = len(passed_vehicles)
        return result


    def passed_time_count(self, fns):
        '''
        passed_time_count
        Get the total time (in minutes) spent on approaching lanes of vehicles that passed the intersection during time interval ∆t after the last action.
        
        :param fns: information of current intersection, including vehicle_trajectory, lane_vehicles, etc
        :return passed_time_count: the total time
        '''
        vehicle_trajectory = fns["vehicle_trajectory"]
        passed_vehicles = self.get_passed_vehicles(fns)
        # for i in passed_vehicles:
        #     print(vehicle_trajectory[i])
        passed_time_count = 0
        for vehicle in passed_vehicles:
            passed_time_count += vehicle_trajectory[vehicle][-2][2]
        return passed_time_count


    def vehicle_map(self, fns):
        '''
        vehicle_map
        Get the location of vehicles in the roadnet.
        
        :param fns: information of current intersection, including vehicle_trajectory, lane_vehicles, etc
        :return mapOfCars: matrix that record location of all vehicles appearing in the road network
        '''
        def vehicle_location_mapper(coordinate, area_length, area_width):
            transformX = math.floor((coordinate[0] + area_length / 2) / grid_width)
            length_width_map = float(area_length) / area_width
            transformY = math.floor((coordinate[1] + area_width / 2) * length_width_map / grid_width)
            length_num_grids = int(area_length / grid_width)
            transformY = length_num_grids - 1 if transformY == length_num_grids else transformY
            transformX = length_num_grids - 1 if transformX == length_num_grids else transformX
            tempTransformTuple = (transformY, transformX)
            return tempTransformTuple

        # the length and width of current intersection
        # TODO get the length and width from RoadNet file
        area_length = 600
        area_width = 600
        grid_width = 4 # hyper parameter, decides the density of the grid

        length_num_grids = int(area_length / grid_width)
        mapOfCars = np.zeros((length_num_grids, length_num_grids))

        vehicle_distance = fns["vehicle_distance"]
        lane_vehicles = fns["lane_vehicles"]
        for lane in self.all_lanes:
            for vehicle in lane_vehicles[lane]:
                distance = vehicle_distance[vehicle]
                vehicle_position = self.get_vehicle_position(distance, lane)  # (double,double),tuple
                if vehicle_position is None:
                    continue
                transform_tuple = vehicle_location_mapper(vehicle_position, area_length, area_width)  # transform the coordinates to location in grid
                mapOfCars[transform_tuple[0], transform_tuple[1]] = 1
        return mapOfCars

    def cur_phase(self,fns):
        '''
        cur_phase
        Get current phase of current intersection.
        
        :param fns: information of current intersection, including vehicle_trajectory, lane_vehicles, etc
        :return cur_phase: current phase
        '''
        cur_phase = fns["phase"]
        return cur_phase

    def generate(self, action_interval=10):
        '''
        generate
        Generate state or reward based on current simulation state.
        
        :param: None
        :return ret: state or reward
        '''
        self.action_interval = action_interval
        self.time = self.world.eng.get_current_time()

        fns = {fn:self.world.get_info(fn) for fn in self.fns}
        ret = [self.result_functions[res](fns) for res in self.targets]

        if self.negative:
            ret = ret * (-1)

        return ret


if __name__ == "__main__":
    from world.world_cityflow import World
    world = World("examples/configs.json", thread_num=1)
    laneVehicle = IntersectionVehicleGenerator(world, world.intersections[0],
                                               ["vehicle_trajectory", "lane_vehicles", "vehicle_distance"],
                                               ["passed_time_count", "passed_count", "vehicle_map"])
    for _ in range(1, 301):
        world.step([_%3])
        ret = laneVehicle.generate()

        if _%10 != 0:
            continue
        print(ret)

