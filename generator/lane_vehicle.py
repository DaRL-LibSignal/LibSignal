import numpy as np
from . import BaseGenerator
from world import world_cityflow, world_sumo


#import world_sumo

class LaneVehicleGenerator(BaseGenerator):
    """
    Generate State or Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    I : Intersection object
    fns : list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay" and "pressure"
    in_only : boolean, whether to compute incoming lanes only
    average : None or str
        None means no averaging
        "road" means take average of lanes on each road
        "all" means take average of all lanes
    negative : boolean, whether return negative values (mostly for Reward)
    """
    def __init__(self, world, I, fns, in_only=False, average=None, negative=False):
        self.world = world
        self.I = I

        # get lanes of intersections
        self.lanes = []
        if in_only:
            roads = I.in_roads
        else:
            roads = I.roads
        # TODO: register it in Registry
        if isinstance(world, world_sumo.World):
            for r in roads:
                if self.world.RIGHT:
                    tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
                else:
                    tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
                self.lanes.append(tmp)
                # TODO: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
        elif isinstance(world, world_cityflow.World):
            for road in roads:
                from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
                self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
        else:
            raise Exception('NOT IMPLEMENTED YET')

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        # calculate result dimensions
        size = sum(len(x) for x in self.lanes)
        if average == "road":
            size = len(roads)
        elif average == "all":
            size = 1
        self.ob_length = len(fns) * size
        if self.ob_length == 3:
            self.ob_length = 4

        self.average = average
        self.negative = negative

    def generate(self):
        results = [self.world.get_info(fn) for fn in self.fns]

        #need modification here

        ret = np.array([])
        for i in range(len(self.fns)):
            result = results[i]

            # pressure returns result of each intersections, so return directly
            if self.I.id in result:
                ret = np.append(ret, result[self.I.id])
                continue
            fn_result = np.array([])

            for road_lanes in self.lanes:
                road_result = []
                for lane_id in road_lanes:
                    road_result.append(result[lane_id])
                if self.average == "road" or self.average == "all":
                    road_result = np.mean(road_result)
                else:
                    road_result = np.array(road_result)
                fn_result = np.append(fn_result, road_result)
            
            if self.average == "all":
                fn_result = np.mean(fn_result)
            ret = np.append(ret, fn_result)
        if self.negative:
            ret = ret * (-1)
        origin_ret = ret
        if len(ret) == 3:
            ret_list = list(ret)
            ret_list.append(0)
            ret = np.array(ret_list)
        if len(ret) == 2:
            ret_list = list(ret)
            ret_list.append(0)
            ret_list.append(0)
            ret = np.array(ret_list)
        return ret

if __name__ == "__main__":
    from world.world_cityflow import World
    world = World("examples/configs.json", thread_num=1)
    laneVehicle = LaneVehicleGenerator(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())
