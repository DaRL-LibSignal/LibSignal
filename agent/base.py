from common.registry import Registry


@Registry.register_model('base')
class BaseAgent(object):
    def __init__(self, world):
        # revise if it is multi-agents in one model
        self.world = world
        self.sub_agents = 1

    def get_ob(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_action(self, ob, phase):
        raise NotImplementedError()
