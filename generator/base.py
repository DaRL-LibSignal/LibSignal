class BaseGenerator(object):
    """
    Generate State or Reward based on current CityFlow state.
    """
    def generate(self):
        raise NotImplementedError()