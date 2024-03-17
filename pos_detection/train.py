from ultralytics import YOLO

def train(
        model : str = "yolov8n-pose.pt",
        data : str = "coco8-pose.yaml",
        epochs : int = 100,
        imgsz : int = 640
        ):
    instance = YOLO(model)
    return instance.train(data=data, epochs=epochs, imgsz=imgsz)
