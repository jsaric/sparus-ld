from data.build_transform import build_train_transform, build_val_transform
from data.spaur_keypoint_dataset import SPAURKeypointDataset


def build_train_dataset(cfg):
    transform = build_train_transform(cfg)
    dataset_name = cfg.DATASETS.TRAIN
    if dataset_name.startswith("SPAUR"):
        res = dataset_name.split("_")[1]
        dataset = SPAURKeypointDataset(
            root_path=f"./datasets/SPAUR/res_{res}",
            split='train',
            transform=transform,
        )
    else:
        raise NotImplementedError
    return dataset


def build_val_dataset(cfg):
    transform = build_val_transform(cfg)
    dataset_name = cfg.DATASETS.VAL
    if dataset_name.startswith("SPAUR"):
        res = dataset_name.split("_")[1]
        dataset = SPAURKeypointDataset(
            root_path=f"./datasets/SPAUR/res_{res}",
            split=cfg.DATASETS.VAL_SPLIT,
            transform=transform,
        )
    else:
        raise NotImplementedError
    return dataset
