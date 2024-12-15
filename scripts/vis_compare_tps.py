from PIL import Image, ImageDraw
import os
from tps.tps_io import TPSFile, TPSImage, TPSPoints
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def vis_compare_tps(tps_pred, tps_gt, image_dir, dot_size=15):
    image_dir = Path(image_dir)
    tps_pred = TPSFile.read_file(tps_pred)
    tps_gt = TPSFile.read_file(tps_gt)
    tps_pred_dict = {tps_example.image: tps_example for tps_example in tps_pred.images}
    tps_gt_dict = {tps_example.image: tps_example for tps_example in tps_gt.images}
    for image_name in tps_pred_dict.keys():
        if image_name not in tps_gt_dict:
            print("Warning: image not found in tps_gt: ", image_name)
            continue
        tps_pred_example = tps_pred_dict[image_name]
        tps_gt_example = tps_gt_dict[image_name]
        image = Image.open(image_dir / image_name)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        scale_gt = tps_gt_example.scale
        scale_pred = tps_pred_example.scale
        keypoints_pred = tps_pred_example.landmarks.points
        keypoints_gt = tps_gt_example.landmarks.points
        keypoints_pred[:, 1] = height - keypoints_pred[:, 1]
        keypoints_gt[:, 1] = height - keypoints_gt[:, 1]
        for i in range(len(keypoints_pred)):
            x_pred, y_pred = keypoints_pred[i]
            x_gt, y_gt = keypoints_gt[i]
            draw.ellipse((x_pred - dot_size, y_pred - dot_size, x_pred + dot_size, y_pred + dot_size), fill='red')
            draw.ellipse((x_gt - dot_size, y_gt - dot_size, x_gt + dot_size, y_gt + dot_size), fill='green')

        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize dataset')
    parser.add_argument('tps_pred', type=str, help='tps pred path')
    parser.add_argument('tps_gt', type=str, help='tps gt path')
    parser.add_argument('image_dir', type=str, help='image dir')
    args = parser.parse_args()
    vis_compare_tps(args.tps_pred, args.tps_gt, args.image_dir)

