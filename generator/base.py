class BaseGenerator(object):
    '''
    Generate state or reward based on current simulation state.
    '''
    def generate(self):
        '''
        generate
        Generate state or reward based on current simulation state. 
        Different types of generators have different methods to implement it.
        
        :param: None
        :return: None
        '''
        raise NotImplementedError()