from .model import YOLOL


def infer(cfg, target=None, save=None):
    model_name = cfg.model.path or cfg.model.name
    model = YOLOL(model_name, mode='infer')
    target = target or cfg.inference.target
    if save is None:
        save = cfg.inference.save
    return model(target, save=save)
