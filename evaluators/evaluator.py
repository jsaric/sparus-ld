class Evaluator:
    def update(self, preds, targets):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class EvaluatorsList(Evaluator):
    def __init__(self, evaluators):
        self.evaluators = evaluators

    def update(self, preds, targets):
        for evaluator in self.evaluators:
            evaluator.update(preds, targets)

    def evaluate(self):
        results = []
        for evaluator in self.evaluators:
            results.append(evaluator.evaluate())
        return results

    def reset(self):
        for evaluator in self.evaluators:
            evaluator.reset()