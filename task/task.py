import logging
from common.registry import Registry


@Registry.register_task('base')
class BaseTask:
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
    def run(self):
        try:
            if Registry.mapping['logger_mapping']['logger_setting'].param['train_model']:
                self.trainer.train()
            if Registry.mapping['logger_mapping']['logger_setting'].param['test_model']:
                self.trainer.test(drop_load=False)
        except RuntimeError as e:
            self._process_error(e)
            raise e
