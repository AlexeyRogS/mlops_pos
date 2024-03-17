import fire
from pos_detection import train, infer, PoseMatcher

def run_matching(model="yolov8n-pose.pt",
                 save_dir="./runs/pose/match",
                 img_path="./test_imgs/game-streamer-tips.jpg"):
    matcher = PoseMatcher(model=model, save_dir=save_dir)
    matcher.get_avatar_pose(img_path)

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "match_pose": run_matching
        }
    )
