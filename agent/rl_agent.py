from . import BaseAgent

class RLAgent(BaseAgent):
    def __init__(self, action_space, ob_generator, reward_generator):
        super().__init__(action_space)
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator

    def get_ob(self):
        return self.ob_generator.generate()

    def get_reward(self):
        reward = self.reward_generator.generate()
        assert len(reward) == 1
        return reward[0]

    def get_action(self, ob):
        return self.action_space.sample()

    def choose(self,**kwargs):
        raise NotImplementedError


class State(object):
    # todo: implement as abstract class

    D_QUEUE_LENGTH = (8,)
    D_NUM_OF_VEHICLES = (8,)
    D_WAITING_TIME = (8,)
    D_MAP_FEATURE = (150,150,1,)
    D_CUR_PHASE = (1,)
    D_NEXT_PHASE = (1,)
    D_TIME_THIS_PHASE = (1,)
    D_IF_TERMINAL = (1,)
    D_HISTORICAL_TRAFFIC = (6,)

