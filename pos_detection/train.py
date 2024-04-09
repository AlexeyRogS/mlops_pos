import subprocess

import lightning as L

from .datasets import COCOL
from .model import YOLOL, EarlyStopping


def train(cfg):
    model_name = cfg.model.path or cfg.model.name
    data = cfg.data.path or cfg.data.name
    print(model_name)
    model = YOLOL(
        model_name,
        data=data,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        batch=cfg.training.batch_size,
        imgsz=cfg.img_size,
        workers=cfg.workers,
    )

    ds = COCOL(
        batch_size=cfg.training.batch_size,
        workers=cfg.workers,
        model=model_name,
        data=data,
    )
    ds.setup()

    trainer = L.Trainer(
        enable_checkpointing=False, logger=False, callbacks=[EarlyStopping()]
    )

    mlflow = subprocess.Popen(
        ["mlflow", "ui", "--host=0.0.0.0", "--backend-store-uri", "file:./runs/mlflow"]
    )
    trainer.fit(model=model, train_dataloaders=ds.train_dataloader())
    mlflow.wait()
