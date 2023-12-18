from pathlib import Path
from vis_utils import draw_keypoints


class VisualizationEvaluator:
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def update(self, preds, targets):
        for i in range(len(targets["image_path"])):
            gt_keypoints = targets["keypoints"][i]
            pred_keypoints = preds["keypoints"][i]
            image_path = Path(targets["image_path"][i])
            vis_image = draw_keypoints(image_path, pred_keypoints.cpu().numpy(), gt_keypoints.cpu().numpy())
            vis_image.save(self.output_dir / Path(targets["image_path"][0]).name)

    def evaluate(self):
        return {"Visualization results directory": str(self.output_dir)}

    def reset(self):
        pass