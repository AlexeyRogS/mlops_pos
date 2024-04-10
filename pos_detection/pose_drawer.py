import cv2
import numpy as np


class PoseDrawer:
    def __init__(self):
        pass

    def draw_pose(self, pose):
        points = pose[0].keypoints.xy[0].detach().cpu().numpy()
        image = np.zeros(pose[0].keypoints.orig_shape, dtype=np.uint8)
        self.draw_head(points, image)
        self.draw_body(points, image)
        return image

    def draw_head(self, points, image):
        width = points[5][0] - points[6][0]
        x, y = (points[1][0] + points[2][0]) / 2, points[0][1]
        cv2.circle(image, (int(x), int(y)), int(width // 4), 255, 6)

        x, y = points[1]
        cv2.circle(image, (int(x), int(y)), 10, 255, 6)

        x, y = points[2]
        cv2.circle(image, (int(x), int(y)), 10, 255, 6)

    def draw_body(self, points, image):
        connect_paths = [[10, 8, 6, 5, 7, 9], [6, 12, 11, 5]]
        for path in connect_paths:
            for i in range(len(path) - 1):
                self._connect(path[i], path[i + 1], points, image)

    def _connect(self, n, m, points, image):
        x1, y1 = points[n]
        x2, y2 = points[m]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), 255, 6)
