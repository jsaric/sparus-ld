import torch
from torchvision.transforms import v2 as transforms_v2


def build_train_transform(cfg):
    transforms_list = [
        transforms_v2.ToImage(),
    ]
    if cfg.INPUT.COLOR_JITTER.ENABLED:
        transforms_list.append(
            transforms_v2.ColorJitter(
                brightness=cfg.INPUT.COLOR_JITTER.BRIGHTNESS,
                contrast=cfg.INPUT.COLOR_JITTER.CONTRAST,
                saturation=cfg.INPUT.COLOR_JITTER.SATURATION,
                hue=cfg.INPUT.COLOR_JITTER.HUE
            )
        )
    transforms_list.append(
        transforms_v2.ToDtype(
            dtype=torch.float32,
            scale=True
        )
    )
    if cfg.INPUT.NORMALIZE:
        transforms_list.append(
            transforms_v2.Normalize(
                mean=torch.tensor(cfg.INPUT.MEAN),
                std=torch.tensor(cfg.INPUT.STD)
            )
        )
    if cfg.INPUT.RANDOM_ROTATION.ENABLED:
        transforms_list.append(
            transforms_v2.RandomRotation(
                cfg.INPUT.RANDOM_ROTATION.ANGLE
            )
        )
    if cfg.INPUT.RANDOM_SHORTER_SIDE.ENABLED:
        transforms_list.append(
            transforms_v2.RandomShortestSize(
                cfg.INPUT.RANDOM_SHORTER_SIDE.MIN_SIZE,
                cfg.INPUT.RANDOM_SHORTER_SIDE.MAX_SIZE,
                antialias=True
            )
        )
    if cfg.INPUT.RANDOM_ERASE.ENABLED:
        transforms_list.append(
            transforms_v2.RandomErasing(
                cfg.INPUT.RANDOM_ERASE.PROBABILITY,
                cfg.INPUT.RANDOM_ERASE.SCALE,
                cfg.INPUT.RANDOM_ERASE.RATIO,
                cfg.INPUT.RANDOM_ERASE.VALUE
            )
        )
    if cfg.INPUT.CROP_TYPE == "center":
        transforms_list.append(
            transforms_v2.CenterCrop(
                cfg.INPUT.SIZE
            )
        )
    elif cfg.INPUT.CROP_TYPE == "random":
        transforms_list.append(
            transforms_v2.RandomCrop(
                cfg.INPUT.SIZE,
                pad_if_needed=True
            )
        )
    return transforms_v2.Compose(transforms_list)


def build_val_transform(cfg):
    transforms_list = [
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(
            dtype=torch.float32,
            scale=True
        )
    ]
    if cfg.INPUT.NORMALIZE:
        transforms_list.append(
            transforms_v2.Normalize(
                mean=torch.tensor(cfg.INPUT.MEAN),
                std=torch.tensor(cfg.INPUT.STD)
            )
        )
    return transforms_v2.Compose(transforms_list)