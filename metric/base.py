class BaseMetric(object):
    def __init__(self, world):
        self.world = world

    def update(self):
        raise NotImplementedError()