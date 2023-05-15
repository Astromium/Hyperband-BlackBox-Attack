import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from evaluators import TorchEvaluator
from typing import Any, Union, Callable, List, Dict
from numpy.typing import NDArray
from constraints.relation_constraint import BaseRelationConstraint
from sklearn.preprocessing import MinMaxScaler
import random


class MainWorker(Worker):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None], alpha: float, beta: float, classifier: Any, x: List, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], int_features: Union[NDArray, None], generate_perturbation: Callable, history: Dict, candidate: NDArray, **kwargs):
        super().__init__(**kwargs)
        self.constraints = constraints
        self.scaler = scaler
        self.alpha = alpha
        self.beta = beta
        self.classifier = classifier
        self.x = x
        self.y = y
        self.eps = eps
        self.distance = distance
        self.features_min_max = features_min_max
        self.int_features = int_features
        self.generate_perturbation = generate_perturbation
        self.history = history
        self.candidate = candidate
        self.evaluator = TorchEvaluator(
            constraints=self.constraints, scaler=self.scaler, alpha=self.alpha, beta=self.beta)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        configuration = set(list(config.values()))
        score, adv, _, _ = self.evaluator.evaluate(
            classifier=self.classifier,
            configuration=configuration,
            budget=int(budget),
            x=self.x,
            y=self.y,
            eps=self.eps,
            distance=self.distance,
            features_min_max=self.features_min_max,
            int_features=self.int_features,
            generate_perturbation=self.generate_perturbation,
            history=self.history,
            candidate=self.candidate
        )

        return ({
                'loss': score,  # remember: HpBandSter always minimizes!
                'info': {'candidate': adv,
                                'budget': budget
                         }

                })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        n = random.randint(40, 63)
        names = [f'feature{i}' for i in range(n)]
        for i in range(n):
            cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(
                names[i], lower=0, upper=62))
        return cs
