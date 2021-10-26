
class BaseAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_ob(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_action(self, ob):
        raise NotImplementedError()