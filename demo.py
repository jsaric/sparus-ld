# python demo.py --config-file configs/r18_spaur_1152x768.yaml --model-weights /path/to/weights.pth --input-image /path/to/image.jpg
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

from config import get_cfg_defaults
from data.build_transform import build_val_transform
from models.build_model import build_model

def resize_image(image, max_size):
    h, w = max_size
    # determine the scale factor to fit the image within the max size
    scale = min(w / image.width, h / image.height)
    # resize the image
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    return image, scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="path to config file")
    parser.add_argument("--model-weights", required=True, type=str, help="path to model weights")
    parser.add_argument("--input-image", required=True, type=str, help="path to input image")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    model = build_model(cfg)
    model.to(device)
    model.load_state_dict(torch.load(args.model_weights, map_location=device)["model"])
    model.eval()

    val_transform = build_val_transform(cfg)
    image_orig = Image.open(args.input_image)
    image, scale = resize_image(image_orig, cfg.INPUT.SIZE)
    image = val_transform({"image": tv_tensors.Image(image)})["image"]

    with torch.no_grad():
        output = model(image.unsqueeze(0))

    detected_keypoints = output["keypoints"][0].cpu().numpy() / scale
    from vis_utils import draw_keypoints
    image_with_keypoints = draw_keypoints(args.input_image, keypoints=detected_keypoints, ps=int(4 / scale), draw_kp_idx=True)
    image_with_keypoints.save("demo_output.jpg")
    np.savetxt("demo_keypoints.txt", detected_keypoints, fmt="%d")

if __name__ == "__main__":
    main()
