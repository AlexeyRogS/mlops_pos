from ultralytics import YOLO

def infer(model : str, input_x : str, **kwargs):
    instance = YOLO(model)
    return instance(input_x, **kwargs)
