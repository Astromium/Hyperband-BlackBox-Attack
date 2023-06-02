from succesive_halving import SuccessiveHalving
from typing import Any, List
from dataclasses import dataclass
from evaluators import Evaluator
from numpy.typing import NDArray
from sampler import Sampler
from worker import Worker
import joblib
from joblib import Parallel, delayed
from adversarial_problem import AdversarialProblem
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.factory import get_crossover, get_mutation, get_problem, get_reference_directions, get_termination, get_sampling
from pymoo.optimize import minimize
from pymoo.util.nds import fast_non_dominated_sort
from pymoo.operators.sampling.rnd import FloatRandomSampling
from utils.tensorflow_classifier import TensorflowClassifier
from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline
from constraints.url_constraints import get_url_relation_constraints
from pymoo.util.nds import fast_non_dominated_sort
from utils.perturbation_generator import generate_perturbation
import numpy as np
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
        
        # Start the RNSGA search for best candidates for each example
        scaler = joblib.load('./ressources/baseline_scaler.joblib')
        classifier = Pipeline(steps=[('preprocessing', scaler), ('model', TensorflowClassifier(load_model(self.classifier_path)))])
        constraints = get_url_relation_constraints()
        final_objectives = []
        sr = 0
        for j, (scores, configurations, candidates) in enumerate(zip(global_scores, global_configs, global_candidates)):
            print(f'Starting Evolution for example {j}')
            k = 0
            best_objectives = []
            best = np.argmin(np.array(scores))
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
                    constraints=constraints, 
                    features_min_max=features_min_max, 
                    scaler=scaler, 
                    configuration=best_config, 
                    int_features=int_features, 
                    eps=self.eps
                )
                
                ref_points = get_reference_directions(
                        "energy", problem.n_obj, self.R, seed=1
                )
                #ref_points = get_reference_directions('uniform', self.R, problem.n_obj)
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

                res = minimize(problem, algorithm, termination=('n_gen', 100))

                optimal_solutions = res.pop.get("X")
                optimal_objectives = res.pop.get("F")
                optimals = []
                for i in range(len(optimal_solutions)):
                    #print(f"Objective values {i}: {optimal_objectives[i]}")
                    if optimal_objectives[i][1] <= self.eps:
                        #print(f"Objective values {i}: {optimal_objectives[i]}")
                        optimals.append(optimal_objectives[i])
                    #scores.append((sum(optimal_objectives[i]), optimal_solutions[i]))
                if len(optimals) > 0:
                    optimals = sorted(optimals, key=lambda k: k[0])
                    print(f'best objective for example {j} {optimals[0]}')
                    final_objectives.append((optimals[0], j))

            '''
            for score, configuration, candidate in zip(scores, configurations, candidates):
                problem = AdversarialProblem(
                    x_clean=self.x[j], 
                    n_var=len(configuration), 
                    y_clean=self.y[j], 
                    classifier=classifier, 
                    constraints=constraints, 
                    features_min_max=features_min_max, 
                    scaler=scaler, 
                    configuration=configuration, 
                    int_features=int_features, 
                    eps=self.eps
                )
                
                ref_points = get_reference_directions(
                        "energy", problem.n_obj, self.R, seed=1
                )
                
                ref_points = get_reference_directions(
                        name="das-dennis", n_partitions=12, n_dim=problem.n_obj, n_points=91, seed=1
                )
                
                #ref_points = get_reference_directions('uniform', self.R, problem.n_obj)
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

                res = minimize(problem, algorithm, termination=('n_gen', 100))

                optimal_solutions = res.pop.get("X")
                optimal_objectives = res.pop.get("F")
                optimals = []
                for i in range(len(optimal_solutions)):
                    
                    adv = np.copy(x)
                    adv[list(configuration)] += np.nan_to_num(optimal_solutions[i])
                    adv = np.clip(adv, features_min_max[0], features_min_max[1])
                    adv = self.fix_feature_types(optimal_solutions[i], adv, int_features, configuration)
                    pred = classifier.predict_proba(adv[np.newaxis, :])[0]
                    print(f'pred of the optimal solution {i} : {pred}')
                    
                    #print(f"Objective values {i}: {optimal_objectives[i]}")
                    if optimal_objectives[i][1] <= self.eps:
                        #print(f"Objective values {i}: {optimal_objectives[i]}")
                        optimals.append(optimal_objectives[i])
                    #scores.append((sum(optimal_objectives[i]), optimal_solutions[i]))
                if len(optimals) > 0:
                    optimals = sorted(optimals, key=lambda k: k[0])
                    best_objectives.append(optimals[0])
                
                fronts = fast_non_dominated_sort.fast_non_dominated_sort(np.array(optimals))
                for front in fronts:
                    print("Front:")
                    for solution in front:
                        print(optimals[solution])
                    print("------------------")
                
            best_objectives = sorted(best_objectives, key=lambda k: k[0])
            print(f'best objectives for example {j} : {best_objectives}')
            if len(best_objectives) > 0:
                final_objectives.append((best_objectives[0], j))
            '''
        
        print(f'final objectives across all examples {final_objectives}')
        cr = (classifier.predict(self.x) == 1).astype('int').sum()
        for i, (obj, j) in enumerate(final_objectives):
            print(f'pred for example {j} : {classifier.predict(self.x[j][np.newaxis, :])[0]}')
            if classifier.predict(self.x[j][np.newaxis, :])[0] != 1:
                print(f'pred inside the if for example {j} {classifier.predict(self.x[j][np.newaxis, :])[0]}')
                continue

            if obj[0] < 0.5 and obj[2] <= 0.00001:
                print(f'adversarial {j} : pred {obj[0]}')
                sr += 1
        print(f'Correct {cr}')
        print(f'Success rate {(sr / cr ) * 100 }%')

        # for b in zip(*results):
        #     scores, configs, candidates = [], [], []
        #     for (s, c, ca) in b:
        #         scores.extend(s)
        #         configs.extend(c)
        #         candidates.extend(ca)
        #     global_scores.append(scores)
        #     global_configs.append(configs)
        #     global_candidates.append(candidates)
        
        #print(f'len global_scores[0] {len(global_scores[0])}')


        #for thread in zip(*results):

        

        # for (scores, configurations, candidates) in results:
        #     all_scores.extend(scores)
        #     all_configurations.extend(configurations)
        #     all_candidates.extend(candidates)
        
        
        #res = processes[0].run()
        #res1 = processes[1].run()
        
        '''
        for p in processes:
            scores, configurations, candidates = p.run()
            with lock:
                all_scores.extend(scores)
                all_configurations.extend(configurations)
                all_candidates.extend(candidates)    
        '''

        '''
        for i in reversed(range(s_max + 1)):
            print(f'\n Starting Hyperband Bracket {i} \n')
            p = Worker(kwargs=params[i])
            n = round((B * (self.downsample ** i)) / (self.R * (i + 1)))
            n = max(n, 1)
            bracket_budget = max(round(self.R / (self.downsample ** i)), 1)
            sh = SuccessiveHalving(
                objective=self.objective,
                classifier=self.classifier, 
                sampler=self.sampler,
                x=self.x,
                y=self.y,
                eps=self.eps,
                dimensions=self.dimensions,
                max_configuration_size=self.max_configuration_size,
                distance=self.distance,
                max_ressources_per_configuration=self.R,
                downsample=self.downsample,
                bracket_budget=bracket_budget,
                n_configurations=n,
                mutables=mutables,
                features_min_max=features_min_max,
                hyperband_bracket=i
            )

            scores, configurations, candidates = p.run()

            all_scores.extend(scores)
            all_configurations.extend(configurations)
            all_candidates.extend(candidates)
            '''
        #print(f'global history {global_history}')
        return global_scores, global_configs, global_candidates, global_history, global_misclassifs, global_viols


