from common.utils import build_index_intersection_map
from common.registry import Registry
from common.utils import load_config_dict
from common.utils import get_output_file_path


class Interface(object):
    def __init__(self):
        pass


@Registry.register_world('graph_setting')
class Graph_World_Interface(Interface):
    def __init__(self, path):
        super(Graph_World_Interface, self).__init__()
        Graph_World_Interface.graph = build_index_intersection_map(path)


@Registry.register_world('traffic_setting')
class Traffic_param_Interface(Interface):
    def __init__(self, path):
        super(Traffic_param_Interface, self).__init__()
        if isinstance(path, dict):
            Traffic_param_Interface.param = path
        elif isinstance(path, str):
            Traffic_param_Interface.param = load_config_dict(path)
        else:
            raise NotImplementedError('traffic setting: input should be path or dictionary')


@Registry.register_model('model_setting')
class ModelAgent_param_Interface(Interface):
    def __init__(self, path):
        super(ModelAgent_param_Interface, self).__init__()
        if isinstance(path, dict):
            ModelAgent_param_Interface.param = path
        elif isinstance(path, str):
            ModelAgent_param_Interface.param = load_config_dict(path)
        else:
            raise NotImplementedError('model setting: input should be path or dictionary')


@Registry.register_logger('output_path')
class Logger_path_Interface(Interface):
    def __init__(self, task, model, prefix):
        super(Logger_path_Interface, self).__init__()
        Logger_path_Interface.path = get_output_file_path(task, model, prefix)


@Registry.register_logger('logger_setting')
class Logger_param_Interface(Interface):
    def __init__(self, path):
        super(Logger_param_Interface, self).__init__()
        if isinstance(path, dict):
            Logger_param_Interface.param = path
        elif isinstance(path, str):
            Logger_param_Interface.param = load_config_dict(path)
        else:
            raise NotImplementedError('logger setting: input should be path or dictionary')


@Registry.register_trainer('trainer_setting')
class Trainer_param_Interface(Interface):
    def __init__(self, path):
        super(Trainer_param_Interface, self).__init__()
        if isinstance(path, dict):
            Trainer_param_Interface.param = path
        elif isinstance(path, str):
            Trainer_param_Interface.param = load_config_dict(path)
        else:
            raise NotImplementedError('task setting: input should be path or dictionary')
