from evaluators.evaluator import Evaluator
import numpy as np


class KeypointSimilarityEvaluator(Evaluator):
    def __init__(self, keypoint_names, sigma, print_results=True):
        super().__init__()
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.sigma = sigma
        self.print_results = print_results
        self.reset()

    def update(self, preds, targets):
        pred_keypoints = preds['keypoints'].cpu().numpy()
        target_keypoints = targets['keypoints'].cpu().numpy()
        distances = np.linalg.norm(pred_keypoints - target_keypoints, axis=-1, ord=2) ** 2
        distances /= (2 * self.sigma ** 2)
        similarities = np.exp(-distances)
        batch_size = distances.shape[0]
        self.similarities += similarities.sum(axis=0)
        self.count += batch_size

    def evaluate(self):
        similarities = self.similarities / self.count
        results = {}
        for i, keypoint_name in enumerate(self.keypoint_names):
            results[f"{keypoint_name}_keypoint_similarity"] = similarities[i]
        results["average_keypoint_similarity"] = similarities.mean()
        if self.print_results:
            self.nice_print(results)
        return results

    def nice_print(self, results):
        print(f"Per keypoint similarity (sigma = {self.sigma}):")
        for i, keypoint_name in enumerate(self.keypoint_names):
            print(f"{i+1}.\t{keypoint_name:<25} {results[f'{keypoint_name}_keypoint_similarity']:0.3f}")
        print(f"Average keypoint similarity (sigma = {self.sigma}): \t {results['average_keypoint_similarity']:0.3f}")

    def reset(self):
        self.similarities = np.zeros(self.num_keypoints)
        self.count = 0
