import fire
from ultralytics import YOLO

def train(
        model : str = "yolov8n-pose.yaml",
        data : str = "coco8-pose.yaml",
        epochs : int = 100,
        imgsz : int = 640
        ):
    instance = YOLO(model)
    return instance.train(data=data, epochs=epochs, imgsz=imgsz)

if __name__ == "__main__":
    fire.Fire(train)
