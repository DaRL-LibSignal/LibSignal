# from .base import BaseGenerator


class IntersectionPhaseGenerator():
    '''
    Generate state or reward based on statistics of intersection phases.

    :param world: World object
    :param I: Intersection object
    :param fns: list of statistics to get, "phase" is needed for result "cur_phase"
    :param targets: list of results to return, currently support "cur_phase": current phase of the intersection (not before yellow phase)
             See section 4.2 of the intelliLight paper[Hua Wei et al, KDD'18] for more detailed description on these targets.
    :param negative: boolean, whether return negative values (mostly for Reward)
    :param time_interval: use to calculate
    '''

    def __init__(self, world, I, fns=("phase"),
                 targets=("cur_phase"), negative=False):
        self.world = world
        self.I = I

        # get cur phase of the intersection
        self.phase = I.current_phase

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns
        self.targets = targets

        self.negative = negative


    def generate(self):
        '''
        generate
        Generate current phase based on current simulation state.
        
        :param: None
        :return ret: result based on current phase
        '''
        return self.I.current_phase


if __name__ == "__main__":
    from world.world_cityflow import World

    world = World("examples/configs.json", thread_num=1)
    laneVehicle = IntersectionPhaseGenerator(world, world.intersections[0],
                                               ["vehicle_trajectory", "lane_vehicles", "vehicle_distance"],
                                               ["passed_time_count", "passed_count", "vehicle_map"])
    for _ in range(1, 301):
        world.step([_ % 3])
        ret = laneVehicle.generate()

        if _ % 10 != 0:
            continue
        print(ret)


