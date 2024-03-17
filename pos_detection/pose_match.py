import os
import cv2
from .infer import infer
from ultralytics import YOLO
from .pose_drawer import PoseDrawer

class PoseMatcher:
    def __init__(self, model, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.drawer = PoseDrawer()
        self.model = model
        self.save_dir = save_dir

    def get_avatar_pose(self, input_x, save=True):
        result = self.drawer.draw_pose(infer(self.model, input_x))
        if save:
            self.save_avatar_pose(result)
        return result
    
    def save_avatar_pose(self, image):
        saved = list(filter(lambda x: x.endswith('.png'),
                    os.listdir(self.save_dir)))
        name = f"{str(len(saved)).zfill(5)}.png"
        cv2.imwrite(os.path.join(self.save_dir, name), image)
