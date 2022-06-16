import numpy as np


class BaseMetric(object):
    def __init__(self, world):
        self.world = world

    def update(self):
        raise NotImplementedError()


class TravelTimeMetric(BaseMetric):
    """
    Calculate average travel time of all vehicles.
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """

    def __init__(self, world):
        super().__init__(world)
        self.world.subscribe(["vehicles", "time"])
        self.vehicle_enter_time = {}
        self.travel_times = []

    def update(self, done=False):
        vehicles = self.world.get_info("vehicles")
        current_time = self.world.get_info("time")

        for vehicle in vehicles:
            if vehicle not in self.vehicle_enter_time:
                self.vehicle_enter_time[vehicle] = current_time

        for vehicle in list(self.vehicle_enter_time):
            if done or vehicle not in vehicles:
                self.travel_times.append(current_time - self.vehicle_enter_time[vehicle])
                del self.vehicle_enter_time[vehicle]

        return np.mean(self.travel_times) if len(self.travel_times) else 0
