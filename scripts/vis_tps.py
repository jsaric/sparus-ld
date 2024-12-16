from PIL import Image, ImageDraw, ImageFont
from tps.tps_io import TPSFile, TPSImage, TPSPoints
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def read_class_name(image_name):
    if "divlje" in image_name:
        return "Wild"
    elif "uzgoj" in image_name:
        return "Farmed"
    elif "tunakavez" in image_name:
        return "Farm-associated"
    else:
        return "Unknown"

def vis_tps(tps_gt, image_dir, dot_size=15):
    image_dir = Path(image_dir)
    tps_gt = TPSFile.read_file(tps_gt)
    tps_gt_dict = {tps_example.image: tps_example for tps_example in tps_gt.images}
    for image_name in tps_gt_dict.keys():
        if image_name not in tps_gt_dict:
            print("Warning: image not found in tps_gt: ", image_name)
            continue
        cls_name = read_class_name(image_name)
        tps_gt_example = tps_gt_dict[image_name]
        image = Image.open(image_dir / image_name)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        scale_gt = tps_gt_example.scale
        keypoints_gt = tps_gt_example.landmarks.points
        keypoints_gt[:, 1] = height - keypoints_gt[:, 1]
        for i in range(len(keypoints_gt)):
            x_gt, y_gt = keypoints_gt[i]
            draw.ellipse((x_gt - dot_size, y_gt - dot_size, x_gt + dot_size, y_gt + dot_size), fill='green')
        # Write class text with large font on middle-top with some margin of the image
        draw.text(
            (width // 2, 50),
            cls_name,
            fill='green',
            anchor='mm',
            font=ImageFont.truetype("arial.ttf", 60),
            align='center'
        )
        image.save("/home/josip/temp/" + image_name)
        # plt.imshow(image)
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize dataset')
    parser.add_argument('tps_gt', type=str, help='tps gt path')
    parser.add_argument('image_dir', type=str, help='image dir')
    args = parser.parse_args()
    vis_tps(args.tps_gt, args.image_dir, dot_size=7)

