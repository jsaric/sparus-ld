from pathlib import Path
from vis_utils import draw_keypoints
import matplotlib.pyplot as plt
import shutil


class VisualizationEvaluator:
    def __init__(self, output_dir, save_heatmaps=False):
        super().__init__()
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.save_heatmaps = save_heatmaps

    def update(self, preds, targets):
        for i in range(len(targets["image_path"])):
            gt_keypoints = targets["keypoints"][i]
            pred_keypoints = preds["keypoints"][i].cpu()
            mse = ((gt_keypoints - pred_keypoints) ** 2).mean().item()
            pred_heatmaps = preds["heatmap"][i].sigmoid()
            image_path = Path(targets["image_path"][i])
            vis_image = draw_keypoints(image_path, pred_keypoints.cpu().numpy(), gt_keypoints.cpu().numpy())
            vis_image.save(self.output_dir / f"{mse:.3f}_{image_path.stem}_keypoints.jpg")
            shutil.copy(image_path, self.output_dir / f"{mse:.3f}_{image_path.stem}_image.jpg")
            if self.save_heatmaps:
                # for j in range(pred_heatmaps.shape[0]):
                #     plt.imsave(self.output_dir / f"{image_path.stem}_heatmap_{j}.jpg", pred_heatmaps[j].cpu().numpy())
                plt.imsave(self.output_dir / f"{mse:.3f}_{image_path.stem}_heatmap_all.jpg", pred_heatmaps.max(0).values.cpu().numpy())
                
    def evaluate(self):
        return {"Visualization results directory": str(self.output_dir)}

    def reset(self):
        pass
