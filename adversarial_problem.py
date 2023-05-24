import numpy as np
from pymoo.core.problem import ElementwiseProblem
from typing import Any, Union, Any, List
from constraints.relation_constraint import BaseRelationConstraint
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.relation_constraint import AndConstraint
from sklearn.preprocessing import MinMaxScaler
import math

NB_OBJECTIVES = 3


def get_nb_objectives():
    return NB_OBJECTIVES


def get_bounds(configuration, features_min, features_max):
    xl, xu = [], []
    for c in configuration:
        xl.append(features_min[c])
        xu.append(features_max[c])
    print(xl, xu)
    return xl, xu


class AdversarialProblem(ElementwiseProblem):
    def __init__(
        self,
        x_clean: np.ndarray,
        n_var: int,
        y_clean: int,
        classifier: Any,
        constraints: Union[List[BaseRelationConstraint], None],
        features_min_max: List,
        scaler: MinMaxScaler,
        configuration: List,
        int_features: np.ndarray,

        norm=2,
    ):
        # Parameters
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.classifier = classifier
        self.constraints = constraints
        self.features_min_max = features_min_max
        self.n_var = n_var
        self.scaler = scaler
        self.configuration = configuration
        self.int_features = int_features
        self.norm = norm

        # Computed attributes
        # xl, xu = get_bounds(configuration=self.configuration,
        #                     features_min=self.features_min_max[0], features_max=self.features_min_max[1])
        xl, xu = min(self.features_min_max[0]), max(self.features_min_max[1])

        super().__init__(
            n_var=self.n_var,
            n_obj=get_nb_objectives(),
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def get_x_clean(self):
        return self.x_clean

    def _obj_misclassify(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.classifier.predict_proba(x)[0][self.y_clean]
        return y_pred

    def _obj_distance(self, x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x_1 - x_2, ord=self.norm)

    def _calculate_constraints(self, x):
        executor = NumpyConstraintsExecutor(
            AndConstraint(self.constraints),
        )
        return executor.execute(x[np.newaxis, :])[0]

    def fix_feature_types(self, adv, x):
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
        if (x - self.xl < 0).sum() > 0:
            print("Lower than lower bound.")

        if (x - self.xu > 0).sum() > 0:
            print("Lower than lower bound.")

        # --- Prepare necessary representation of the samples

        # Retrieve original representation
        x_adv = np.copy(self.x_clean)
        x_adv[list(self.configuration)] += x

        x_adv = np.clip(
            x_adv, self.features_min_max[0], self.features_min_max[1])

        x_adv = self.fix_feature_types(x_adv, x)

        obj_misclassify = self._obj_misclassify(x_adv[np.newaxis, :])
        obj_distance = self._obj_distance(
            self.scaler.transform(x_adv[np.newaxis, :])[
                0], self.scaler.transform(self.x_clean[np.newaxis, :])[0]
        )

        obj_constraints = self._calculate_constraints(x_adv)

        F = [obj_misclassify, obj_distance, obj_constraints]

        # --- Output
        out["F"] = np.column_stack(F)
