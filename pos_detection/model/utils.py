import torch
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer


class YoloOpt(Optimizer):
    def __init__(self, params, yolo_train):
        super().__init__(params, dict())
        self.yolo_train = yolo_train

    def step(self, *args, **kwargs):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.yolo_train.scaler.unscale_(self.yolo_train.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            self.yolo_train.model.parameters(), max_norm=10.0
        )  # clip gradients
        self.yolo_train.scaler.step(self.yolo_train.optimizer)
        self.yolo_train.scaler.update()
        self.yolo_train.optimizer.zero_grad()
        if self.yolo_train.ema:
            self.yolo_train.ema.update(self.yolo_train.model)


class EarlyStopping(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        should_stop = pl_module.yolo_train.stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
