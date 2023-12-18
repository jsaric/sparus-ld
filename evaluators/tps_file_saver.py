from evaluators.evaluator import Evaluator
import numpy as np
from tps.tps_io import TPSImage, TPSFile, TPSPoints


class TPSFileSaver(Evaluator):
    def __init__(self, output_dir, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.output_dir = output_dir
        self.reset()

    def update(self, preds, targets):
        pred_keypoints = preds['keypoints'].cpu().numpy()
        scales = targets['scale'].cpu().numpy()
        for i in range(len(pred_keypoints)):
            h, w = targets['image'][i].shape[-2:]
            keypoints = pred_keypoints[i]
            scale = scales[i] / self.scaling_factor
            keypoints[:, 1] = h - keypoints[:, 1]
            landmarks = TPSPoints((keypoints * self.scaling_factor).astype(np.int32))
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
        TPSFile(images=self.tps_preds).write_to_file(self.output_dir / "tps_predictions.TPS")
        return {}

    def reset(self):
        self.tps_preds = []
