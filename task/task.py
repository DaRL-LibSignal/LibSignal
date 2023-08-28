import logging
from common.registry import Registry


@Registry.register_task('base')
class BaseTask:
    '''
    Register BaseTask, currently support TSC task.
    '''
    def __init__(self, trainer):
        self.trainer = trainer

    def run(self):
        raise NotImplementedError

    def _process_error(self, e):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
        ):
            for name, parameter in self.trainer.agents.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

@Registry.register_task("tsc")
class TSCTask(BaseTask):
    '''
    Register Traffic Signal Control task.
    '''
    def run(self):
        '''
        run
        Run the whole task, including training and testing.

        :param: None
        :return: None
        '''
        try:
            if Registry.mapping['model_mapping']['setting'].param['train_model']:
                self.trainer.train()
            if Registry.mapping['model_mapping']['setting'].param['test_model']:
                self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e

@Registry.register_task("tscfx")
class TSCFXTask(BaseTask):
    '''
    Register Traffic Signal Control task.
    '''
    def run(self):
        '''
        run
        Run the whole task, including training and testing.

        :param: None
        :return: None
        '''
        try:
            if Registry.mapping['model_mapping']['setting'].param['train_model']:
                self.trainer.train()
            if Registry.mapping['model_mapping']['setting'].param['test_model']:
                self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e