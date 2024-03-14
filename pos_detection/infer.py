import fire
from ultralytics import YOLO

def infer(model : str, input_x : str, **kwargs):
    instance = YOLO(model)
    return instance(input_x, **kwargs)

if __name__ == "__main__":
    fire.Fire(infer)