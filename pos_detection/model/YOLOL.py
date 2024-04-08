import math
import time
import torch
import numpy as np
import lightning as L
from .utils import YoloOpt
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from torch import distributed as dist
from ultralytics.utils import RANK, DEFAULT_CFG, LOGGER, colorstr

class YOLOL(L.LightningModule):
    def __init__(self, model, mode='train', cfg=DEFAULT_CFG, **kwargs):
        super().__init__()
        self.checkpoint_callback=False
        self.automatic_optimization=False
        self.model = YOLO(model)
        self.args = {"model": model, "task": "pose", **kwargs}
        if mode == "train":
            self.yolo_train = self.model.task_map["pose"]['trainer'](overrides=self.args)
        self.args = get_cfg(cfg, self.args)

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def on_train_start(self):
        if torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0
        self.yolo_train.world_size = world_size
        self.yolo_train._setup_train(world_size)
        self.yolo_train.nb = len(self.yolo_train.train_loader)  # number of batches
        self.yolo_train.nw = max(
            round(self.yolo_train.args.warmup_epochs * self.yolo_train.nb),
            100) if self.yolo_train.args.warmup_epochs > 0 else -1  # warmup iterations
        self.yolo_train.last_opt_step = -1
        self.yolo_train.epoch_time = None
        self.yolo_train.epoch_time_start = time.time()
        self.yolo_train.train_time_start = time.time()
        self.yolo_train.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.yolo_train.args.imgsz} train, {self.yolo_train.args.imgsz} val\n'
            f'Using {self.yolo_train.train_loader.num_workers*(world_size or 1)}'
            ' dataloader workers\n'
            f"Logging results to {colorstr('bold', self.yolo_train.save_dir)}\n"
            f'Starting training for ' + \
            (f"{self.yolo_train.args.time} hours..."
             if self.yolo_train.args.time else f"{self.yolo_train.epochs} epochs...")
        )
        if self.yolo_train.args.close_mosaic:
            base_idx = (self.yolo_train.epochs -
                        self.yolo_train.args.close_mosaic) * self.yolo_train.nb
            self.yolo_train.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        self.yolo_train.epoch = self.yolo_train.start_epoch

    def training_step(self, batch, i):
        self.yolo_train.run_callbacks("on_train_batch_start")
        # Warmup
        last_opt_step = self.yolo_train.last_opt_step
        world_size = self.yolo_train.world_size
        epoch = self.yolo_train.epoch
        ni = self.yolo_train.ni = i + self.yolo_train.nb * self.yolo_train.epoch
        if ni <= self.yolo_train.nw:
            xi = [0, self.yolo_train.nw]  # x interp
            self.yolo_train.accumulate = \
                max(1,
                    int(np.interp(ni, xi, 
                    [1, self.yolo_train.args.nbs / self.yolo_train.batch_size]).round()))
            for j, x in enumerate(self.yolo_train.optimizer.param_groups):
                # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    ni, xi,
                    [self.yolo_train.args.warmup_bias_lr if j == 0 else 0.0,
                     x["initial_lr"] * self.yolo_train.lf(epoch)]
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(ni, xi,
                    [self.yolo_train.args.warmup_momentum, self.yolo_train.args.momentum])

        # Forward
        with torch.cuda.amp.autocast(self.yolo_train.amp):
            batch = self.yolo_train.preprocess_batch(batch)
            self.yolo_train.loss, self.yolo_train.loss_items = self.yolo_train.model(batch)
            if RANK != -1:
                self.yolo_train.loss *= world_size
            self.yolo_train.tloss = (
                (self.yolo_train.tloss * i + self.yolo_train.loss_items) / (i + 1)
                if self.yolo_train.tloss is not None
                else self.yolo_train.loss_items
            )

        # Backward
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.yolo_train.scaler.scale(self.yolo_train.loss)
        self.manual_backward(loss)

        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
        if ni - last_opt_step >= self.yolo_train.accumulate:
            self.yolo_train.last_opt_step = last_opt_step = ni
            opt.step()
            # Timed stopping
            if self.yolo_train.args.time:
                self.yolo_train.stop = \
                    (time.time() - 
                     self.yolo_train.train_time_start) > (self.yolo_train.args.time * 3600)
                if RANK != -1:  # if DDP training
                    broadcast_list = [self.yolo.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    self.yolo_train.stop = broadcast_list[0]

        # Log
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        loss_len = self.yolo_train.tloss.shape[0] if len(self.yolo_train.tloss.shape) else 1
        losses = self.yolo_train.tloss if loss_len > 1 else torch.unsqueeze(self.yolo_train.tloss, 0)
        if RANK in {-1, 0}:
            self.yolo_train.run_callbacks("on_batch_end")
            if self.yolo_train.args.plots and ni in self.yolo_train.plot_idx:
                self.yolo_train.plot_training_samples(batch, ni)
        self.yolo_train.lr = {f"lr/pg{ir}": x["lr"] for ir, x in
                              enumerate(self.yolo_train.optimizer.param_groups)}  # for loggers
        self.yolo_train.run_callbacks("on_train_batch_end")
        return loss
    
    def on_train_epoch_end(self):
        epoch = self.yolo_train.epoch
        if RANK in {-1, 0}:
            final_epoch = epoch + 1 >= self.yolo_train.epochs
            self.yolo_train.ema.update_attr(
                self.yolo_train.model,
                include=["yaml", "nc", "args", "names", "stride", "class_weights"]
            )

            # Validation
            if self.yolo_train.args.val or final_epoch or \
                self.yolo_train.stopper.possible_stop or self.yolo_train.stop:
                self.yolo_train.metrics, self.yolo_train.fitness = self.yolo_train.validate()
            self.yolo_train.metrics = {
                        **self.yolo_train.label_loss_items(self.yolo_train.tloss),
                        **self.yolo_train.metrics, **self.yolo_train.lr
                    }
            self.yolo_train.save_metrics(metrics = self.yolo_train.metrics)
            for key, value in self.yolo_train.metrics.items():
                self.log(key,  value)
            self.yolo_train.stop |= self.yolo_train.stopper(epoch + 1,
                                    self.yolo_train.fitness) or final_epoch
            if self.yolo_train.args.time:
                self.yolo_train.stop |= \
                    (time.time() - 
                    self.yolo_train.train_time_start) > (self.yolo_train.args.time * 3600)

            # Save model
            if self.yolo_train.args.save or final_epoch:
                self.yolo_train.save_model()
                self.yolo_train.run_callbacks("on_model_save")

        # Scheduler
        t = time.time()
        self.yolo_train.epoch_time = t - self.yolo_train.epoch_time_start
        self.yolo_train.epoch_time_start = t
        if self.yolo_train.args.time:
            mean_epoch_time = \
            (t - self.yolo_train.train_time_start) / (epoch - self.yolo_train.start_epoch + 1)
            self.yolo_train.epochs = self.yolo_train.args.epochs = \
                math.ceil(self.yolo_train.args.time * 3600 / mean_epoch_time)
            self.yolo_train._setup_scheduler()
            self.yolo_train.scheduler.last_epoch = self.yolo_train.epoch  # do not move
            self.yolo_train.stop |= epoch >= self.yolo_train.epochs  # stop if exceeded epochs
        self.yolo_train.run_callbacks("on_fit_epoch_end")
        torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

        # Early Stopping
        if RANK != -1:  # if DDP training
            broadcast_list = [self.yolo_train.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            self.yolo_train.stop = broadcast_list[0]
        self.yolo_train.epoch += 1
    
    def on_train_end(self):
        epoch = self.yolo_train.epoch
        if RANK in {-1, 0}:
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.yolo_train.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.yolo_train.train_time_start) / 3600:.3f} hours."
            )
            self.yolo_train.final_eval()
            if self.yolo_train.args.plots:
                self.yolo_train.plot_metrics()
            self.yolo_train.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.yolo_train.run_callbacks("teardown")

    def configure_optimizers(self):
        return YoloOpt(self.model.model.parameters(), self.yolo_train)
