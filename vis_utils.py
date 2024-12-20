from PIL import Image, ImageDraw


def draw_keypoints(image_path, keypoints=None, gt_keypoints=None, metrics=None, ps=3, draw_kp_idx=False):
    fish_image = Image.open(image_path)
    draw = ImageDraw.Draw(fish_image)
    num_keypoints = len(keypoints) if keypoints is not None else len(gt_keypoints)

    for i in range(num_keypoints):
        if gt_keypoints is not None:
            gt_keypoint = gt_keypoints[i]
            draw.ellipse([tuple(gt_keypoint - ps), tuple(gt_keypoint + ps)], fill=(0, 255, 0))
        if keypoints is not None:
            keypoint = keypoints[i]
            draw.ellipse([tuple(keypoint - ps), tuple(keypoint + ps)], fill=(255, 0, 0))
            if draw_kp_idx:
                draw.text((keypoint[0], keypoint[1] + 10), str(i + 1), fill=(255, 0, 0), font_size=ps * 5)
        if metrics is not None:
            similarity_string = f'{metrics[i]:.2f}'
            bbox = draw.textbbox((keypoint[0] + 10, keypoint[1]), similarity_string)
            draw.rectangle(bbox, fill='black')
            draw.text((keypoint[0] + 10, keypoint[1]), similarity_string, fill='white')
    return fish_image