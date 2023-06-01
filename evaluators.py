from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union, Callable, List, Dict
from numpy.typing import NDArray
from scipy.special import softmax
from constraints.relation_constraint import BaseRelationConstraint
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.relation_constraint import AndConstraint
from sklearn.preprocessing import MinMaxScaler
from utils.mutation_generator import generate_mutations
from adversarial_problem import AdversarialProblem
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.factory import get_crossover, get_mutation, get_problem, get_reference_directions, get_termination, get_sampling
from pymoo.optimize import minimize
from pymoo.util.nds import fast_non_dominated_sort
import numpy as np
import math


@dataclass
class Evaluator(ABC):

    @abstractmethod
    def evaluate_mutations(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None]):
        pass

    @abstractmethod
    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None]):
        pass


class TfEvaluator(Evaluator):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None], alpha: float, beta: float):
        super().__init__()
        self.constraints = constraints
        self.constraint_executor = NumpyConstraintsExecutor(
            AndConstraint(constraints)) if constraints is not None else None
        self.scaler = scaler
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], generate_perturbation: Callable):
        score = 0.0
        best_score = math.inf
        best_adversarial = None
        adv = np.array(x)
        for _ in range(budget):
            perturbation = generate_perturbation(shape=np.array(
                configuration).shape, eps=eps, distance=distance)
            # perturbation = np.random.randn(*np.array(configuration).shape)
            adv[list(configuration)] += perturbation

            # projecting into the Lp-ball
            norm = 2 if distance == 'l2' else np.inf
            dist = np.linalg.norm(adv - x, ord=norm)
            if dist > eps:
                adv = x + (adv - x) * eps / dist
            # clipping into min-max values
            if features_min_max:
                adv = np.clip(adv, features_min_max[0], features_min_max[1])

            pred = softmax(classifier.predict(adv[np.newaxis, :]))
            violations = 0.0
            if self.constraints:
                if self.scaler:
                    adv_rescaled = self.scaler.inverse_transform(
                        np.array(adv)[np.newaxis, :])
                violations = self.constraint_executor.execute(adv_rescaled)[0]
            score = self.alpha * pred[0][y] + self.beta * violations
            # score = self.alpha + self.beta * violations
            if score < best_score:
                best_score = score
                best_adversarial = adv

        return round(best_score, 3), best_adversarial


class TorchEvaluator(Evaluator):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None], alpha: float, beta: float):
        super().__init__()
        self.constraints = constraints
        self.constraint_executor = NumpyConstraintsExecutor(
            AndConstraint(constraints)) if constraints is not None else None
        self.scaler = scaler
        self.alpha = alpha
        self.beta = beta

    def process_mutant(self, xm, x, y, features_min_max, int_features, classifier):
        xm_scaled = self.scaler.transform(xm[np.newaxis, :])[0]
        x_scaled = self.scaler.transform(x[np.newaxis, :])[0]
        dist = np.linalg.norm(xm_scaled - x_scaled)
        if dist > 0.2:
            xm_scaled = x_scaled + (xm_scaled - x_scaled) * 0.2 / dist
            xm = self.scaler.inverse_transform(xm_scaled[np.newaxis, :])[0]
        # clipping
        xm = np.clip(xm, features_min_max[0], features_min_max[1])
        # casting
        xm[int_features] = xm[int_features].astype('int')

        pred = classifier.predict_proba(xm[np.newaxis, :])[0]
        violations = self.constraint_executor.execute(xm[np.newaxis, :])[0]
        return (self.alpha * pred[y] + self.beta * violations, np.copy(xm))

    def evaluate_mutations(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], int_features: Union[NDArray, None], generate_perturbation: Callable, history: Dict, candidate: NDArray):
        mutations = generate_mutations(
            pop_size=budget, features_min_max=features_min_max, x=x)
        scores = [self.process_mutant(xm=mutant, x=x, y=y, features_min_max=features_min_max,
                                      int_features=int_features, classifier=classifier) for mutant in mutations]
        scores = sorted(scores, key=lambda k: k[0])
        return round(scores[0][0], 3), scores[0][1], 0, 0

    def fix_feature_types(self, perturbation, adv, int_features, configuration):
        for i, c in enumerate(configuration):
            if c in int_features:
                adv[c] = math.ceil(
                    adv[c]) if perturbation[i] < 0 else math.floor(adv[c])
        return adv

    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], int_features: Union[NDArray, None], generate_perturbation: Callable, history: Dict, candidate: NDArray):
        '''
        scores = []
        misclassifs = [0] * budget
        viols = [0] * budget
        perturbations = [
            generate_perturbation(configuration=configuration,
                                  features_min=features_min_max[0],
                                  features_max=features_min_max[1], x=x)
            for _ in range(budget)
        ]
        perturbations = np.array(perturbations)
        problem = AdversarialProblem(x_clean=x, n_var=len(configuration), y_clean=y, classifier=classifier, constraints=self.constraints,
                                     features_min_max=features_min_max, scaler=self.scaler, configuration=configuration, int_features=int_features, norm=2)

        ref_points = get_reference_directions(
                "energy", problem.n_obj, 100, seed=1
            )

        # get_sampling('real_random')

        algorithm = RNSGA3(  # population size
            n_offsprings=100,  # number of offsprings
            sampling=perturbations,  # use the provided initial population
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True,
            ref_points=ref_points,
            pop_per_ref_point=1
        )

        res = minimize(problem, algorithm, termination=('n_gen', 300))

        optimal_solutions = res.pop.get("X")
        optimal_objectives = res.pop.get("F")


        for i in range(len(optimal_solutions)):
            #print("Solution:", optimal_solutions[i])

            adv = np.copy(x)
            adv[list(configuration)] += np.nan_to_num(optimal_solutions[i])
            adv = np.clip(adv, features_min_max[0], features_min_max[1])
            adv = self.fix_feature_types(optimal_solutions[i], adv, int_features, configuration)
            pred = classifier.predict_proba(adv[np.newaxis, :])[0]
            print(f'pred of the optimal solution {i} : {pred}')

            print(f"Objective values {i}: {optimal_objectives[i]}")
            scores.append((sum(optimal_objectives[i]), optimal_solutions[i]))

        return round(scores[0][0], 3), scores[0][1], misclassifs[0], viols[0]
        '''

        score = 0.0
        best_score = math.inf
        best_adversarial = None
        scores = [0] * budget
        misclassif = [0] * budget
        viols = [0] * budget
        if candidate is None:
            adv = np.copy(x)
        else:
            adv = np.copy(candidate)
        for i in range(budget):
            # perturbation = generate_perturbation(shape=np.array(configuration).shape, eps=eps, distance=distance)
            perturbation = generate_perturbation(
                configuration=configuration, features_min=features_min_max[0], features_max=features_min_max[1], x=x)
            adv[list(configuration)] += perturbation

            adv_scaled = self.scaler.transform(adv[np.newaxis, :])[0]
            x_scaled = self.scaler.transform(x[np.newaxis, :])[0]
            dist = np.linalg.norm(adv_scaled - x_scaled, ord=distance)
            if dist > eps:
                adv_scaled = x_scaled + (adv_scaled - x_scaled) * eps / dist
                # transform back to pb space
                adv = self.scaler.inverse_transform(
                    adv_scaled[np.newaxis, :])[0]

            # clipping
            adv = np.clip(adv, features_min_max[0], features_min_max[1])
            # casting
            adv = self.fix_feature_types(
                perturbation=perturbation, adv=adv, int_features=int_features, configuration=configuration)

            pred = classifier.predict_proba(adv[np.newaxis, :])[0]
            if self.constraints:
                violations = self.constraint_executor.execute(adv[np.newaxis, :])[
                    0]
                scores[i] = (self.alpha * pred[y] + self.beta *
                             violations, np.copy(adv))
            else:
                scores[i] = (self.alpha * pred[y], np.copy(adv))

            '''
            if score < best_score:
                print('New score')
                dist1 = np.linalg.norm(x - adv)
                print(f'dist of adv {dist1}')
                best_score = score
                best_adversarial = np.copy(adv)
                dist1 = np.linalg.norm(x - best_adversarial)
                print(f'dist after assignement {dist1}')
            '''

            # dist = np.linalg.norm(best_adversarial - x, ord=distance)
            # print(f'dist of best {dist}')
            # if dist > eps:
            # best_adversarial = x + (best_adversarial - x) * eps / dist
        scores = sorted(scores, key=lambda k: k[0])
        return round(scores[0][0], 3), scores[0][1], misclassif[0], viols[0]


class SickitEvaluator(Evaluator):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None], alpha: float, beta: float):
        super().__init__()
        self.constraints = constraints
        self.constraint_executor = NumpyConstraintsExecutor(
            AndConstraint(constraints)) if constraints is not None else None
        self.scaler = scaler
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], generate_perturbation: Callable):
        score = 0.0
        adv = np.array(x)
        best_score = math.inf
        best_adversarial = None
        for _ in range(budget):
            perturbation = generate_perturbation(shape=np.array(
                configuration).shape, eps=eps, distance=distance)
            adv[list(configuration)] += perturbation

            # projecting into the Lp-ball
            norm = 2 if distance == 'l2' else np.inf
            dist = np.linalg.norm(self.scaler.transform(adv[np.newaxis, :])[
                                  0] - self.scaler.transform(x[np.newaxis, :]), ord=norm)
            if dist > eps:
                adv = x + (adv - x) * eps / dist

            # clipping into min-max feature values
            if features_min_max:
                adv = np.clip(adv, features_min_max[0], features_min_max[1])

            violations = 0.0
            if self.constraints:
                violations = self.constraint_executor.execute(adv[np.newaxis, :])[
                    0]

            pred = classifier.predict_proba(adv[np.newaxis, :])
            score = self.alpha * pred[0][y] + self.beta * violations

            if score < best_score:
                best_score = score
                best_adversarial = adv

        return round(best_score, 3), best_adversarial
