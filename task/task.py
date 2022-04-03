import logging
import os
from common.registry import Registry

class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        if self.config["checkpoint"] is not None:
            self.trainer.load_checkpoint(self.config["checkpoint"])

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self):
        raise NotImplementedError

    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

@Registry.register_task("tsc")
class TSCTask(BaseTask):
    def run(self):
        try:
            self.trainer.train()
            self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e