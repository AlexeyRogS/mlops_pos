import hydra
from omegaconf import DictConfig
from ultralytics import settings

from pos_detection import PoseMatcher, infer, train


settings.update({'mlflow': True})


def run_matching(cfg):
    matcher = PoseMatcher(cfg)
    matcher.get_avatar_pose(cfg.matching.target)


@hydra.main(config_path="cfg", config_name='config', version_base="1.3")
def run(cfg: DictConfig):
    # We simply print the configuration
    print(cfg)
    if cfg['mode'] == 'train':
        train(cfg)
    elif cfg['mode'] == 'infer':
        infer(cfg)
    elif cfg['mode'] == 'match_pose':
        run_matching(cfg)
    else:
        raise NotImplementedError(f"Wrong mode {cfg.mode}")


if __name__ == "__main__":
    run()
