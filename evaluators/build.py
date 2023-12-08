from evaluators.evaluator import EvaluatorsList


def build_evaluators(cfg, val_loader):
    evaluators_names = cfg.EVALUATORS
    if len(evaluators_names) == 0:
        return None
    evaluators = []
    for evaluator_name in evaluators_names:
        if evaluator_name == "euclidean_distance_evaluator":
            from evaluators.euclidean_distance_evaluator import EuclideanDistanceEvaluator
            evaluator = EuclideanDistanceEvaluator(val_loader.dataset.KEYPOINT_NAMES)
        elif evaluator_name == "visualization_evaluator":
            from evaluators.visualization_evaluator import VisualizationEvaluator
            evaluator = VisualizationEvaluator(cfg)
        else:
            raise NotImplementedError(f"Unknown evaluator: {evaluator_name}")
        evaluators.append(evaluator)

    if len(evaluators) == 1:
        return evaluators[0]
    else:
        return EvaluatorsList(evaluators)