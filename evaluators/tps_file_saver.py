from pathlib import Path

from evaluators.evaluator import Evaluator
import numpy as np
from tps.tps_io import TPSImage, TPSFile, TPSPoints
from PIL import Image


class TPSFileSaver(Evaluator):
    def __init__(self, output_file, scaling_factor=None, img_orig_res_folder=None):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.output_file = output_file
        self.img_orig_res_folder = img_orig_res_folder
        if img_orig_res_folder is not None:
            self.img_orig_res_folder = Path(self.img_orig_res_folder)
        if self.scaling_factor is None:
            assert self.img_orig_res_folder is not None, "If scaling_factor is not provided, img_orig_res_folder must be provided"
        self.reset()

    def update(self, preds, targets):
        pred_keypoints = preds['keypoints'].cpu().numpy()
        scales = targets['scale'].cpu().numpy()
        for i in range(len(pred_keypoints)):
            h, w = targets['image'][i].shape[-2:]

            if self.scaling_factor is None:
                w_orig, h_orig = Image.open(self.img_orig_res_folder / targets["image_name"][i]).size
                scaling_factor = w_orig / w
            else:
                scaling_factor = self.scaling_factor

            keypoints = pred_keypoints[i]
            scale = scales[i] / scaling_factor
            keypoints[:, 1] = h - keypoints[:, 1]
            landmarks = TPSPoints((keypoints * scaling_factor).astype(np.int32))

            tps_example = TPSImage(
                image=targets['image_name'][i],
                landmarks=landmarks,
                scale=scale,
                id_number=targets['id_number'][i],
                comment=None,
                curves=None
            )
            self.tps_preds.append(tps_example)

    def evaluate(self):
        TPSFile(images=self.tps_preds).write_to_file(self.output_file)
        return {}

    def reset(self):
        self.tps_preds = []
