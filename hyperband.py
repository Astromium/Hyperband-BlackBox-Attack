from succesive_halving import SuccessiveHalving
from typing import Any, List
from dataclasses import dataclass
from evaluators import Evaluator
from numpy.typing import NDArray
from sampler import Sampler
from worker import Worker
import joblib
from joblib import Parallel, delayed
from mlc.datasets.dataset_factory import get_dataset
from sklearn.pipeline import Pipeline
from ml_wrappers import wrap_model
from adversarial_problem import AdversarialProblem
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.factory import get_crossover, get_mutation, get_problem, get_reference_directions, get_termination, get_sampling
from pymoo.optimize import minimize
from pymoo.util.nds import fast_non_dominated_sort
from pymoo.operators.sampling.rnd import FloatRandomSampling
from tensorflow.keras.models import load_model
from constraints.relation_constraint import AndConstraint
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.lcld_constraints import get_relation_constraints
from utils.tensorflow_classifier import LcldTensorflowClassifier
import numpy as np
import pandas as pd
import os
import math



print(f'Hello {os.getpid()} from Hyperband')

def run_worker(args):
    sh = SuccessiveHalving(**args)
    all_results = sh.run()
    return all_results

@dataclass
class Hyperband():
    objective: Evaluator
    classifier_path: str
    x: NDArray
    y: NDArray
    sampler: Sampler
    eps: float
    dimensions: int
    max_configuration_size: int
    R: int
    downsample: int
    distance: str
    seed: int


    def generate(self, mutables=None, features_min_max=None, int_features=None):
        if self.downsample <= 1:
            raise ValueError('Downsample must be > 1')
        
        all_scores, all_configurations, all_candidates = [], [], []
        # Number of Hyperband rounds
        s_max = math.floor(math.log(self.R, self.downsample))
        B = self.R * (s_max + 1)
        print(f'Hyperband brackets {s_max + 1}')
        params = [ 
            {'objective': self.objective, 
             'classifier_path': self.classifier_path, 
             'sampler': self.sampler, 
             'x': self.x, 'y': self.y, 
             'eps': self.eps, 
             'dimensions': self.dimensions, 
             'max_configuration_size': self.max_configuration_size, 
             'distance': self.distance, 
             'max_ressources_per_configuration': self.R,
             'downsample': self.downsample,
             'mutables': mutables,
             'features_min_max': features_min_max,
             'int_features': int_features,
             'n_configurations': max(round((B * (self.downsample ** i)) / (self.R * (i + 1))), 1),
             'bracket_budget': max(round(self.R / (self.downsample ** i)), 1),
             'seed': self.seed,
             'hyperband_bracket': i,
             'R': self.R
            } 
            for i in reversed(range(s_max + 1)) 
        ]
        
        '''
        p = Pool(os.cpu_count())
        results = p.map(run_worker, params)
        p.close()
        p.join()
        '''
        
        results = Parallel(n_jobs=s_max+1, verbose=0, backend='multiprocessing', prefer='processes')(delayed(run_worker)(params[i]) for i in range(s_max + 1))
        global_scores = []
        global_configs = []
        global_candidates = []
        global_history = []
        global_misclassifs = []
        global_viols = []
        for i in range(self.x.shape[0]):
            scores, configs, candidates, history, history_misclassif, history_viols = [], [], [], [], [], []
            for th in results:
                scores.extend(th[i][0])
                configs.extend(th[i][1])
                candidates.extend(th[i][2])
                history.append(th[i][3])
                history_misclassif.append(th[i][4])
                history_viols.append(th[i][5])
            global_scores.append(scores)
            global_configs.append(configs)
            global_candidates.append(candidates)
            global_history.extend(history)
            global_misclassifs.extend(history_misclassif)
            global_viols.extend(history_viols)

        # start the rnsga search 
        # Start the RNSGA search for best candidates for each example
        scaler = joblib.load('./ressources/lcld_preprocessor.joblib')
        classifier = Pipeline(steps=[('preprocessing', scaler), ('model',
                                                                 LcldTensorflowClassifier(load_model(self.classifier_path)))])
        ds = get_dataset('lcld_v2_iid')
        splits = ds.get_splits()
        x, y = ds.get_x_y()

        categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']

    #x[categorical] = x[categorical].astype(str)

        numerical = [col for col in x.columns if col not in categorical]
        num_indices = [x.columns.get_loc(col) for col in numerical]
        col_order = list(numerical) + list(categorical)
        x = x[col_order]
        feature_names = x.columns.to_list()
        cat_indices = [x.columns.get_loc(col) for col in categorical]
        print(f'cat indices {cat_indices}')

        x = x.to_numpy()
        x_test, y_test = x[splits['test']], y[splits['test']]
        charged_off = np.where(y_test == 1)[0]
        x_charged_off, y_charged_off = x_test[charged_off], y_test[charged_off] 
        #x_clean = scaler.transform(x_clean)

        #model_pipeline = Pipeline(steps=[('preprocessing', preprocessing_pipeline), ('model', rf)])
        #metadata = ds.get_metadata(only_x=True)
        metadata = pd.read_csv('./ressources/lcld_v2_metadata_transformed.csv')
        min_constraints = metadata['min'].to_list()
        print(f'min constraints {min_constraints}')
        min_constraints = list(map(float, min_constraints))
        max_constraints = metadata['max'].to_list()
        print(f'max constraints {max_constraints}')
        max_constraints = list(map(float, max_constraints))
        feature_types = metadata['type'].to_list()
        print(f'features types {feature_types}')
        mutables = metadata.index[metadata['mutable'] == True].tolist()
        print(f'mutables {mutables}')
        int_features = np.where(np.array(feature_types) == 'int')[0]
        cat_features = np.where(np.array(feature_types) == 'cat')[0]
        int_features = list(int_features) + list(cat_features)
        features_min_max = (min_constraints, max_constraints)
        constraints = get_relation_constraints()
        executor = NumpyConstraintsExecutor(AndConstraint(constraints), feature_names=feature_names)

        final_objectives = []
        sr = 0
        misclassifs = 0
        viols = 0
        for j, (scores, configurations, candidates) in enumerate(zip(global_scores, global_configs, global_candidates)):
            print(f'Starting Evolution for example {j}')
            k = 0
            best_objectives = []
            fronts = fast_non_dominated_sort.fast_non_dominated_sort(
                np.array(scores))
            best = fronts[0][0]
            best_adv = candidates[best]
            if classifier.predict(self.x[j][np.newaxis, :])[0] != self.y[j] and classifier.predict(best_adv[np.newaxis, :])[0] == self.y[j]:
                sr += 1
                continue
            else:
                best_config = configurations[best]

                problem = AdversarialProblem(
                    x_clean=self.x[j],
                    n_var=len(best_config),
                    y_clean=self.y[j],
                    classifier=classifier,
                    constraints_executor=executor,
                    features_min_max=features_min_max,
                    scaler=scaler,
                    configuration=best_config,
                    int_features=int_features,
                    eps=self.eps
                )

                ref_points = get_reference_directions(
                    "energy", problem.n_obj, self.R, seed=1
                )
                # ref_points = get_reference_directions('uniform', self.R, problem.n_obj)
                # get_sampling('real_random')
                algorithm = RNSGA3(  # population size
                    n_offsprings=100,  # number of offsprings
                    sampling=FloatRandomSampling(),  # use the provided initial population
                    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                    mutation=get_mutation("real_pm", eta=20),
                    eliminate_duplicates=True,
                    ref_points=ref_points,
                    pop_per_ref_point=1,
                )

                res = minimize(problem, algorithm, termination=('n_gen', 300))

                optimal_solutions = res.pop.get("X")
                optimal_objectives = res.pop.get("F")
                optimals = []
                misclassifs += np.any(np.array([obj[0]
                                      for obj in optimal_objectives]) < 0.5)
                tolerance = 0.01
                viols += np.any(np.array([obj[2]
                                for obj in optimal_objectives]) <= tolerance)
                for i in range(len(optimal_solutions)):
                    # print(f"Objective values {i}: {optimal_objectives[i]}")
                    if optimal_objectives[i][1] <= self.eps:
                        # print(f"Objective values {i}: {optimal_objectives[i]}")
                        optimals.append(optimal_objectives[i])
                    # scores.append((sum(optimal_objectives[i]), optimal_solutions[i]))
                if len(optimals) > 0:
                    optimals = sorted(optimals, key=lambda k: k[0])
                    print(f'best objective for example {j} {optimals[0]}')
                    final_objectives.append((optimals[0], j))

        print(f'final objectives across all examples {final_objectives}')
        cr = (classifier.predict(self.x) == 1).astype('int').sum()
        for i, (obj, j) in enumerate(final_objectives):
            print(
                f'pred for example {j} : {classifier.predict(self.x[j][np.newaxis, :])[0]}')
            if classifier.predict(self.x[j][np.newaxis, :])[0] != 1:
                print(
                    f'pred inside the if for example {j} {classifier.predict(self.x[j][np.newaxis, :])[0]}')
                continue

            if obj[0] < 0.5 and obj[2] <= 0.01:
                print(f'adversarial {j} : pred {obj[0]}')
                sr += 1
        print(f'Correct {cr}')
        print(f'Success rate {(sr / cr ) * 100 }%')
        print(f'Misclassifs {(misclassifs / cr ) * 100 }%')
        print(f'Violations {(viols / cr ) * 100 }%')
        history = {(self.R, self.eps): {'M': (misclassifs / cr) *
                                        100, 'C&M': (sr / cr) * 100, 'C': (viols / cr) * 100}}
        print(f'history {history}')


        return global_scores, global_configs, global_candidates, global_history, global_misclassifs, global_viols


