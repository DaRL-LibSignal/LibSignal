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