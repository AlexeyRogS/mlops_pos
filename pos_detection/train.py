import subprocess
import lightning as L
from .datasets import COCOL
# from ultralytics import YOLO
from .model import YOLOL, EarlyStopping
# from lightning.pytorch.loggers import MLFlowLogger


# def train(
#         model : str = "yolov8n-pose.pt",
#         data : str = "coco8-pose.yaml",
#         epochs : int = 100,
#         imgsz : int = 640
#         ):
#     instance = YOLO(model)
#     return instance.train(data=data, epochs=epochs, imgsz=imgsz)

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
        workers=cfg.workers
    )


    ds = COCOL(
        batch_size=cfg.training.batch_size,
        workers=cfg.workers,
        model=model_name,
        data=data,
    )
    ds.setup()

    trainer = L.Trainer(
        enable_checkpointing=False, logger=False,
        callbacks=[EarlyStopping()]
    )
    
    mlflow = subprocess.Popen("mlflow ui --backend-store-uri file:./runs/mlflow")
    trainer.fit(model=model, train_dataloaders=ds.train_dataloader())
    mlflow.wait()


# if __name__ == "__main__":

#     ds = COCOL(
#         batch_size=32,
#         workers=1,
#         model="yolov8n-pose.pt",
#         data="cfg/datasets/coco17-pose.yaml",
#     )

#     model = YOLOL(
#         "yolov8n-pose.pt",
#         data="cfg/datasets/coco17-pose.yaml",
#         epochs=100,
#         patience=100,
#         batch=10,
#         imgsz=640,
#         workers=1
#     )
#     print(model.yolo_train.testset)

#     ds.setup()
#     print(ds.args)

    
#     # model = YOLOL("yolov8n-pose.pt")
#     # ds = COCOL()
#     # ds.setup()

#     # # print(trainer.should_stop)
#     # print(model.args.data)



#     # trainer.fit(model=model, train_dataloaders=ds.train_dataloader())
    
#     # print(model("test_imgs/game-streamer-tips.jpg"))
    
#     batch = next(iter((ds.train_dataloader())))
#     print(batch['img'])
#     # result = model(batch)
#     # print(result)


