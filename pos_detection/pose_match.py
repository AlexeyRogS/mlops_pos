import os
import cv2
from .infer import infer
from .pose_drawer import PoseDrawer

class PoseMatcher:
    def __init__(self, cfg):
        self.drawer = PoseDrawer()
        self.cfg = cfg
        self.save_dir = cfg.matching.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def get_avatar_pose(self, input_x, save=True):
        result = self.drawer.draw_pose(infer(self.cfg, target=input_x, save=False))
        if save:
            self.save_avatar_pose(result)
        return result
    
    def save_avatar_pose(self, image):
        saved = list(filter(lambda x: x.endswith('.png'),
                    os.listdir(self.save_dir)))
        name = f"{str(len(saved)).zfill(5)}.png"
        cv2.imwrite(os.path.join(self.save_dir, name), image)
