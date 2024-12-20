import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import tv_tensors
from pathlib import Path
from tps.tps_io import TPSFile
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from .keypoint_utils import generate_channel_heatmap, get_keypoints_from_heatmap
import matplotlib.pyplot as plt
from vis_utils import draw_keypoints

def _classification_from_image_name(image_name):
    if 'divlje' in image_name:
        return 'divlja'
    elif 'uzgoj' in image_name:
        return 'uzgoj'
    elif 'tunakavez' in image_name:
        return 'tunakavez'
    else:
        return 'nepoznato'


class SPAURKeypointDataset(Dataset):
    KEYPOINT_NAMES_SIMPLE = [
        'mouth',
        'forehead',
        'topFinLeft',
        'topFinMiddle',
        'topFinRight',
        'tailTop',
        'tailEnd',
        'tailBottom',
        'bottomRearFinRight',
        'bottomRearFinLeft',
        'bottomFrontFin',
        'gillBottom',
        'gillMiddle',
        'gillTop',
        'sideFinTop',
        'sideFinBottom',
        'eyeLeft',
        'eyeRight'
    ]

    KEYPOINT_NAMES = [
        "anterior tip of snout at the upper jaw",
        "vertical point above the most anterior point in the eye",
        "anterior insertion of the dorsal fin",
        "last spiny ray of the dorsal fin",
        "posterior insertion of the dorsal fin",
        "dorsal point at the least depth of the caudal peduncle",
        "posterior body extremity",
        "ventral point at the least depth of the caudal peduncle",
        "posterior insertion of the anal fin",
        "anterior insertion of the anal fin",
        "insertion of the pelvic fin",
        "ventral tip of the insertion of the operculum on the lateral profile",
        "point of maximum extension of the operculum on the lateral profile",
        "anterior extremity of the lateral line on the head profile",
        "dorsal insertion of the pectoral fin",
        "ventral insertion of the pectoral fin",
        "the most anterior point in the eye",
        "the most posterior point in the eye",
    ]

    def __init__(self, root_path, split, transform=None):
        self.root_path = Path(root_path)
        self.split = split
        self.image_dir = self.root_path / split
        tps_file = TPSFile.read_file(self.root_path / f"{split}_landmark_configuration.TPS")
        self.tps_items = tps_file.images
        self.train = split == 'train'
        self.transform = transform
        print(f"Loaded dataset with split {self.split} that contains {len(self)} images")

    def __len__(self):
        return len(self.tps_items)

    def __getitem__(self, item):
        tps_item = self.tps_items[item]
        image_name = tps_item.image
        image = Image.open(self.image_dir / image_name)
        w, h = image.size
        keypoints = tps_item.landmarks.points
        keypoints[:, 1] = h - keypoints[:, 1]

        keypoints_tv = tv_tensors.BoundingBoxes(
            np.hstack([keypoints, np.zeros_like(keypoints, dtype=np.float32)]),
            format=tv_tensors.BoundingBoxFormat.CXCYWH,
            canvas_size=(h, w)
        )
        image_tv = tv_tensors.Image(image)

        datum = self.transform({
            "image": image_tv,
            "keypoints": keypoints_tv,
            "class": _classification_from_image_name(image_name),
            "image_path": (self.image_dir / image_name).__str__(),
            "scale": -1 if tps_item.scale is None else tps_item.scale,
            "id_number": tps_item.id_number,
            "image_name": image_name,
        })
        datum["keypoints"] = datum["keypoints"][:, :2].float()
        return datum


if __name__ == '__main__':
    dataset = SPAURKeypointDataset(
        root_path='/mnt/sdb1/datasets/izor22data-new/res_1152x768/',
        split='val',
        transform=transforms_v2.Compose([
            transforms_v2.ColorJitter(brightness=(0.5, 1.3), contrast=0, saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
            transforms_v2.RandomErasing(p=0.1, scale=(0.01, 0.1), ratio=(0.7, 1.3), value=(0, 0, 0)),
            transforms_v2.RandomShortestSize(min_size=(700, 780), max_size=1200),
            transforms_v2.RandomRotation(3),
            transforms_v2.CenterCrop((768, 1152))
        ])
    )
    for i in range(len(dataset)):
        data = dataset[i]
        print(data["keypoints"] - get_keypoints_from_heatmap(data['keypoints_heatmap']))
        image_with_keypoints = draw_keypoints(Image.fromarray(data['image'].permute(1, 2, 0).numpy()),
                                              data['keypoints'].numpy()[:, :2], ps=3)
        plt.subplots(2, 1)
        plt.subplot(2, 1, 1)
        plt.imshow(np.array(image_with_keypoints))
        plt.subplot(2, 1, 2)
        plt.imshow(data['keypoints_heatmap'].max(0)[0].cpu().numpy())
        plt.show()
