import torch
from copy import deepcopy
from datetime import datetime
from torch.optim import Optimizer
from ultralytics.utils import __version__
from lightning.pytorch.callbacks import Callback
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16


class YoloOpt(Optimizer):
    def __init__(self, params, yolo_train):
        super().__init__(params, dict())
        self.yolo_train = yolo_train
    
    def step(self, *args, **kwargs):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.yolo_train.scaler.unscale_(self.yolo_train.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.yolo_train.model.parameters(), max_norm=10.0)  # clip gradients
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


# class SaveModel(Callback):
#     def on_train_epoch_end(self, trainer, pl_module):
#         self._save_movel(trainer, pl_module.yolo_train)

#     def _save_movel(self, trainer, yolo_train):
#         """Save model training checkpoints with additional metadata."""
#         import io

#         import pandas as pd  # scope for faster 'import ultralytics'

#         # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
#         buffer = io.BytesIO()
#         torch.save(
#             {
#                 "epoch": yolo_train.epoch,
#                 "best_fitness": yolo_train.best_fitness,
#                 "model": None,  # resume and final checkpoints derive from EMA
#                 "ema": deepcopy(yolo_train.ema.ema).half(),
#                 "updates": yolo_train.ema.updates,
#                 "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(yolo_train.optimizer.state_dict())),
#                 "train_args": vars(yolo_train.args),  # save as dict
#                 "train_metrics": {**yolo_train.metrics, **{"fitness": yolo_train.fitness}},
#                 "train_results": {k.strip(): v for k, v in pd.read_csv(yolo_train.csv).to_dict(orient="list").items()},
#                 "date": datetime.now().isoformat(),
#                 "version": __version__,
#                 "license": "AGPL-3.0 (https://ultralytics.com/license)",
#                 "docs": "https://docs.ultralytics.com",
#             },
#             buffer,
#         )
#         serialized_ckpt = buffer.getvalue()  # get the serialized content to save

#         # Save checkpoints
#         yolo_train.last.write_bytes(serialized_ckpt)  # save last.pt
#         if yolo_train.best_fitness == yolo_train.fitness:
#             yolo_train.best.write_bytes(serialized_ckpt)  # save best.pt
#         if (yolo_train.save_period > 0) and (yolo_train.epoch > 0) and (yolo_train.epoch % yolo_train.save_period == 0):
#             (yolo_train.wdir / f"epoch{yolo_train.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
