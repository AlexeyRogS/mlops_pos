import lightning as L
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class COCOL(L.LightningDataModule):
    def __init__(self, cfg=DEFAULT_CFG, batch_size: int = 32, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.args = {"task": "pose", "batch": batch_size, **kwargs}
        self.args = get_cfg(cfg, self.args)

    def setup(self, stage: str = "train"):
        self.trainset, self.valset = self.get_dataset()
        self.train_loader = self.get_dataloader(
            self.trainset, batch_size=self.batch_size, rank=-1, mode="train"
        )
        self.val_loader = self.get_dataloader(
            self.valset, batch_size=self.batch_size, rank=-1, mode="val"
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset"""
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val"
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(
            dataset, batch_size, workers, shuffle, rank
        )  # return dataloader

    def get_dataset(self):
        if self.args.task == "classify":
            data = check_cls_dataset(self.args.data)
        elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
            "detect",
            "segment",
            "pose",
            "obb",
        }:
            data = check_det_dataset(self.args.data)
            if "yaml_file" in data:
                self.args.data = data["yaml_file"]
        self.data = data
        return data["train"], data.get("val") or data.get("test")
