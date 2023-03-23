from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union, Callable, List
from numpy.typing import NDArray
from scipy.special import softmax
from constraints.relation_constraint import BaseRelationConstraint
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.relation_constraint import AndConstraint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math

@dataclass
class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None]):
        pass


class TfEvaluator(Evaluator):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None], alpha: float, beta: float):
        super().__init__()
        self.constraints = constraints
        self.constraint_executor = NumpyConstraintsExecutor(AndConstraint(constraints)) if constraints is not None else None
        self.scaler = scaler
        self.alpha = alpha
        self.beta = beta
    
    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], generate_perturbation: Callable):
        score = 0.0
        best_score = math.inf
        best_adversarial = None
        adv = np.array(x)
        for _ in range(budget):
            perturbation = generate_perturbation(shape=np.array(configuration).shape, eps=eps, distance=distance)
            #perturbation = np.random.randn(*np.array(configuration).shape)
            adv[list(configuration)] += perturbation

            # projecting into the Lp-ball
            norm = 2 if distance == 'l2' else np.inf
            dist = np.linalg.norm(adv - x, ord=norm)
            if dist > eps:
                adv = x + (adv - x) * eps / dist
            # clipping into min-max values
            if features_min_max:
                adv = np.clip(adv, features_min_max[0], features_min_max[1])

            #pred = softmax(classifier.predict(adv[np.newaxis, :]))
            violations = 0.0
            if self.constraints:
                if self.scaler:
                    adv_rescaled = self.scaler.inverse_transform(np.array(adv)[np.newaxis, :])
                violations = self.constraint_executor.execute(adv_rescaled)[0]
            #score = self.alpha * pred[0][y] + self.beta * violations 
            score = self.alpha + self.beta * violations
            if score < best_score:
                best_score = score
                best_adversarial = adv

        return round(best_score, 3), best_adversarial
    
class SickitEvaluator(Evaluator):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None], alpha: float, beta: float):
        super().__init__()
        self.constraints = constraints
        self.constraint_executor = NumpyConstraintsExecutor(AndConstraint(constraints)) if constraints is not None else None
        self.scaler = scaler
        self.alpha = alpha
        self.beta = beta
    
    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], generate_perturbation: Callable):
        score = 0.0
        adv = np.array(x)
        best_score = math.inf
        best_adversarial = None
        for _ in range(budget):
            perturbation = generate_perturbation(shape=np.array(configuration).shape, eps=eps, distance=distance)
            adv[list(configuration)] += perturbation

            # projecting into the Lp-ball
            norm = 2 if distance == 'l2' else np.inf
            dist = np.linalg.norm(adv - x, ord=norm)
            if dist > eps:
                adv = x + (adv - x) * eps / dist
            
            #clipping into min-max feature values
            if features_min_max:
                adv = np.clip(adv, features_min_max[0], features_min_max[1])
            
            violations = 0.0
            if self.constraints:
                violations = self.constraint_executor.execute(adv[np.newaxis, :])[0]
            
            pred = classifier.predict_proba(adv[np.newaxis, :])
            score = self.alpha * pred[0][y] + self.beta * violations

            if score < best_score:
                best_score = score
                best_adversarial = adv
        
        return round(best_score, 3), best_adversarial
            