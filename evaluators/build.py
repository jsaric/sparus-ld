from pathlib import Path

from evaluators.evaluator import EvaluatorsList


def build_evaluators(cfg, val_loader):
    evaluators_names = cfg.EVALUATION.EVALUATORS
    if len(evaluators_names) == 0:
        return None
    evaluators = []
    for evaluator_name in evaluators_names:
        if evaluator_name == "euclidean_distance_evaluator":
            from evaluators.euclidean_distance_evaluator import EuclideanDistanceEvaluator
            evaluator = EuclideanDistanceEvaluator(val_loader.dataset.KEYPOINT_NAMES)
        elif evaluator_name == "visualization_evaluator":
            from evaluators.visualization_evaluator import VisualizationEvaluator
            evaluator = VisualizationEvaluator(Path(cfg.OUTPUT_DIR) / "visualization_results")
        elif evaluator_name == "keypoint_similarity_evaluator":
            from evaluators.keypoint_similarity_evaluator import KeypointSimilarityEvaluator
            evaluator = KeypointSimilarityEvaluator(val_loader.dataset.KEYPOINT_NAMES, cfg.LOSS.SIGMA)
        elif evaluator_name == "tps_file_saver":
            from evaluators.tps_file_saver import TPSFileSaver
            evaluator = TPSFileSaver(Path(cfg.OUTPUT_DIR), cfg.EVALUATION.TPS_SAVER_SCALING_FACTOR)
        else:
            raise NotImplementedError(f"Unknown evaluator: {evaluator_name}")
        evaluators.append(evaluator)

    if len(evaluators) == 1:
        return evaluators[0]
    else:
        return EvaluatorsList(evaluators)