from .infer import infer
from ultralytics import YOLO
from pose_drawer import PoseDrawer

class PoseMatcher:
    def __init__(self, model):
        self.instance = YOLO(model)
        self.drawer = PoseDrawer()

    def get_avatar_pose(self, input_x):
        return self.drawer(self.instance(input_x))
