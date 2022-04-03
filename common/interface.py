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
class Traffic_path_Interface(Interface):
    def __init__(self, path):
        super(Traffic_path_Interface, self).__init__()
        Traffic_path_Interface.param = load_config_dict(path)


@Registry.register_model('model_setting')
class ModelAgent_param_Interface(Interface):
    def __init__(self, path):
        super(ModelAgent_param_Interface, self).__init__()
        ModelAgent_param_Interface.param = load_config_dict(path)


@Registry.register_logger('output_path')
class Logger_path_Interface(Interface):
    def __init__(self, task, model, prefix):
        super(Logger_path_Interface, self).__init__()
        Logger_path_Interface.path = get_output_file_path(task, model, prefix)


@Registry.register_logger('logger_setting')
class Logger_param_Interface(Interface):
    def __init__(self, path):
        super(Logger_param_Interface, self).__init__()
        Logger_param_Interface.param = load_config_dict(path)


@Registry.register_task('task_setting')
class Task_param_Interface(Interface):
    def __init__(self, path):
        super(Task_param_Interface, self).__init__()
        Task_param_Interface.param = load_config_dict(path)

