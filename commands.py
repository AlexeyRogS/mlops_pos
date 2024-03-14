import fire
from pos_detection import train, infer

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
        }
    )
