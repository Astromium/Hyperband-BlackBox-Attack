import numpy as np
from pymoo.core.problem import ElementwiseProblem, Problem
from typing import Any, Union, Any, List
from constraints.relation_constraint import BaseRelationConstraint
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.relation_constraint import AndConstraint
from sklearn.preprocessing import MinMaxScaler
import math

NB_OBJECTIVES = 3


def get_nb_objectives():
    return NB_OBJECTIVES


def get_bounds(configuration, features_min, features_max, x):
    xl, xu = [], []
    for c in configuration:
        xl.append(features_min[c] - x[c])
        xu.append(features_max[c] - x[c])
    return xl, xu


class AdversarialProblem(Problem):
    def __init__(
        self,
        x_clean: np.ndarray,
        n_var: int,
        y_clean: int,
        classifier: Any,
        constraints_executor: Any,
        features_min_max: List,
        scaler: MinMaxScaler,
        configuration: List,
        int_features: np.ndarray,
        eps: float,
        norm=2,
    ):
        # Parameters
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.classifier = classifier
        self.constraints_executor = constraints_executor
        self.features_min_max = features_min_max
        self.n_var = n_var
        self.scaler = scaler
        self.configuration = configuration
        self.int_features = int_features
        self.eps = eps
        self.norm = norm

        # Computed attributes
        # xl, xu = min(self.features_min_max[0]), 100#max(self.features_min_max[1])
        # print(f'xl, xu {xl}, {xu}')
        xl, xu = get_bounds(configuration=configuration,
                            features_min=self.features_min_max[0], features_max=self.features_min_max[1], x=self.x_clean)
        # print(f'xl == xu {np.array((xl == xu)).astype("int").sum()}')
        # print(f'xu {xu}')
        # xl, xu = 0.1, 0.9
        super().__init__(
            n_var=self.n_var,
            n_obj=get_nb_objectives(),
            n_constr=3,
            xl=xl,
            xu=xu,
        )

    def get_x_clean(self):
        return self.x_clean

    def _obj_misclassify(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.classifier.predict_proba(x)[:, self.y_clean]
        return y_pred

    def _obj_distance(self, x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(x_1 - x_2, ord=self.norm)
        return dist

    def _calculate_constraints(self, x):
        return self.constraints_executor.execute(x)

    def fix_feature_types(self, adv, x):
        # print(f'adv before {adv}')
        adv = np.nan_to_num(x=adv, copy=True)
        # print(f'adv after {adv}')
        for i, c in enumerate(self.configuration):
            if c in self.int_features:
                # print(f'adv {adv}')
                # print(f'adv[{c}] {adv[c]}')
                adv[c] = math.ceil(
                    adv[c]) if x[i] < 0 else math.floor(adv[c])
        return adv

    def _evaluate(self, x, out, *args, **kwargs):

        # print("Evaluate")

        # Sanity check
        '''
        if (x - self.xl < 0).sum() > 0:
            print("Lower than lower bound.")

        if (x - self.xu > 0).sum() > 0:
            print("Lower than lower bound.")
        '''
        # --- Prepare necessary representation of the samples

        # Retrieve original representation
        x_adv = np.tile(self.x_clean, (x.shape[0], 1))
        x = np.nan_to_num(x=x, copy=True)
        x_adv[:, list(self.configuration)] += x

        x_adv_fixed = np.array([self.fix_feature_types(x1, x2)
                               for x1, x2 in zip(x_adv, x)])

        obj_misclassify = self._obj_misclassify(x_adv_fixed)
        obj_distance = np.array([self._obj_distance(self.scaler.transform(x_adv.reshape(
            1, -1)), self.scaler.transform(self.x_clean.reshape(1, -1))) for x_adv in x_adv_fixed])

        obj_constraints = self._calculate_constraints(x_adv_fixed)

        g1 = obj_distance - 2*self.eps
        g2 = obj_misclassify - 0.5
        g3 = obj_constraints - 0.0001

        F = [obj_misclassify, obj_distance, obj_constraints]
        G = [g1, g2, g3]

        # --- Output
        out["F"] = np.column_stack(F)
        out["G"] = np.column_stack(G)
