

# Borrowed from open catalyst project
"""
Registry is central source of truth. Inspired from Redux's concept of
global store, Registry maintains mappings of various information to unique
keys. Special functions in registry can be used as decorators to register
different kind of classes.
"""


class Registry:
    mapping = {
        'task_mapping': {},
        'dataset_mapping': {},
        'model_mapping': {},
        'logger_mapping': {},
        'world_mapping': {},
        'trainer_mapping': {}
    }

    @classmethod
    def register_world(cls, name):
        def wrap(f):
            cls.mapping['world_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(f):
            cls.mapping['model_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_logger(cls, name):
        def wrap(f):
            cls.mapping['logger_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_task(cls, name):
        def wrap(f):
            cls.mapping['task_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_trainer(cls, name):
        def wrap(f):
            cls.mapping['trainer_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_dataset(cls, name):
        def wrap(f):
            cls.mapping['dataset_mapping'][name] = f
            return f
        return wrap


Registry()
