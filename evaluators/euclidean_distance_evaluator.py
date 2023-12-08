from evaluators.evaluator import Evaluator
import numpy as np


class EuclideanDistanceEvaluator(Evaluator):
    def __init__(self, keypoint_names):
        super().__init__()
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.reset()

    def update(self, preds, targets):
        pred_keypoints = preds['keypoints'].cpu().numpy()
        target_keypoints = targets['keypoints'].cpu().numpy()
        scales = targets['scale'].cpu().numpy()
        distances = np.linalg.norm(pred_keypoints - target_keypoints, axis=-1)
        batch_size = distances.shape[0]
        self.pixel_distance += distances.sum(axis=0)
        self.count += batch_size
        valid_scale_indices = (scales != -1).nonzero()
        metric_distances = distances[valid_scale_indices] * scales[valid_scale_indices].reshape(-1, 1)
        self.metric_distance += metric_distances.reshape(-1, self.num_keypoints).sum(axis=0)
        self.metric_count += len(valid_scale_indices)

    def evaluate(self):
        pixel_distance = self.pixel_distance / self.count
        metric_distance = self.metric_distance / self.metric_count
        results = {}
        for i, keypoint_name in enumerate(self.keypoint_names):
            results[f"{keypoint_name}_pixel_distance"] = pixel_distance[i]
            results[f"{keypoint_name}_metric_distance"] = metric_distance[i]
        results["pixel_distance"] = pixel_distance.mean()
        results["metric_distance"] = metric_distance.mean()
        return results

    def reset(self):
        self.pixel_distance = np.zeros(self.num_keypoints)
        self.metric_distance = np.zeros(self.num_keypoints)
        self.count = 0
        self.metric_count = 0
