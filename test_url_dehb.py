import time
import numpy as np
import warnings
from evaluators import TorchEvaluator
from typing import Any, Union, Callable, List, Dict
from numpy.typing import NDArray
from constraints.relation_constraint import BaseRelationConstraint
from sklearn.preprocessing import MinMaxScaler
import random
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import pandas as pd
from constraints.relation_constraint import AndConstraint
from constraints.url_constraints import get_url_relation_constraints
from constraints.constraints_executor import NumpyConstraintsExecutor
from sklearn.pipeline import Pipeline
from utils.sr_calculators import TorchCalculator
from utils.tensorflow_classifier import TensorflowClassifier
from sampler import Sampler
from evaluators import TorchEvaluator
from tensorflow.keras.models import load_model
from ml_wrappers import wrap_model
from utils.perturbation_generator import generate_perturbation
from adversarial_problem import AdversarialProblem
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.factory import get_crossover, get_mutation, get_problem, get_reference_directions, get_termination, get_sampling
from pymoo.optimize import minimize
from pymoo.util.nds import fast_non_dominated_sort
from pymoo.operators.sampling.rnd import FloatRandomSampling
import joblib


seed = 42
np.random.seed(seed)
warnings.filterwarnings('ignore')
history = {}

min_budget, max_budget = 2, 81

import ConfigSpace as CS


def create_search_space(seed=42):
    """Parameter space to be optimized --- contains the hyperparameters
    """
    cs = CS.ConfigurationSpace()
    #n = random.randint(40, 63)
    names = [f'feature{i}' for i in range(63)]
    for i in range(63):
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(
            names[i], lower=0, upper=1))
    
    return cs

cs = create_search_space(seed)
params = cs.sample_configuration(1)
print(f'cs {params}')
dimensions = len(cs.get_hyperparameters())



def target_function(config, budget, **kwargs):
    
    constraints = kwargs['constraints']
    scaler = kwargs['scaler']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    classifier = kwargs['classifier']
    x_clean = kwargs['x_clean']
    y = kwargs['y']
    eps = kwargs['eps']
    distance = kwargs['distance']
    features_min_max = kwargs['features_min_max']
    int_features = kwargs['int_features']
    generate_perturbation = kwargs['generate_perturbation']
    history = kwargs['history']
    candidate = kwargs['candidate']
    evaluator = TorchEvaluator(
        constraints=constraints, scaler=scaler, alpha=alpha, beta=beta)
        
    
    configuration = list(config.values())
    c = np.where(np.array(configuration) == 1)[0]
    score, adv, _, _ = evaluator.evaluate(
            classifier=classifier,
            configuration=c,
            budget=int(budget),
            x=x_clean,
            y=y,
            eps=eps,
            distance=distance,
            features_min_max=features_min_max,
            int_features=int_features,
            generate_perturbation=generate_perturbation,
            history=history,
            candidate=candidate
        )
    
    result = {
        "fitness": score,  # DE/DEHB minimizes
        "cost": 1,
        "info": {
            "adv": adv,
            "budget": budget
        }
    }
    return result

from dehb import DEHB

x_clean = np.load('./ressources/baseline_X_test_candidates.npy')
y_clean = np.load('./ressources/baseline_y_test_candidates.npy')
metadata = pd.read_csv('./ressources/url_metadata.csv')
min_constraints = metadata['min'].to_list()[:63]
max_constraints = metadata['max'].to_list()[:63]
feature_types = metadata['type'].to_list()[:63]
int_features = np.where(np.array(feature_types) == 'int')[0]
features_min_max = (min_constraints, max_constraints)
constraints = get_url_relation_constraints()
executor = NumpyConstraintsExecutor(AndConstraint(constraints))
scaler = preprocessing_pipeline = joblib.load(
    './ressources/baseline_scaler.joblib')

model_tf = TensorflowClassifier(
    load_model(r'ressources\baseline_nn.model')
)
model_pipeline = Pipeline(
        steps=[('preprocessing', preprocessing_pipeline), ('model', model_tf)])

dimensions = x_clean.shape[1]
BATCH_SIZE = 100#x_clean.shape[0]
eps = 0.2
downsample = 3
sampler = Sampler()
distance = 2
classifier_path = './ressources/baseline_nn.model'
seed = 202374
np.random.seed(seed)
success_rates_l2 = []
exec_times_l2 = []

R_values = [81]
R = 81
history_dict = dict()

host = 'localhost'
shared_directory = '.'
run_id = '0'
min_budget = 1
max_budget = 81
n_iterations = 3
eta = 2

import os

dehb = DEHB(
    f=target_function, 
    cs=cs, 
    dimensions=dimensions, 
    min_budget=min_budget, 
    max_budget=max_budget,
    n_workers=1,
    output_path="./temp",
    eta=eta
)
import timeit
alpha=1.0
beta=1.0

candidates = []
start = timeit.default_timer()
final_objectives = []
kk = 0
for j in range(BATCH_SIZE):
    dehb.reset()
    trajectory, runtime, history = dehb.run(
        #total_cost=10,
        brackets=3,
        verbose=False,
        save_intermediate=False,
        # parameters expected as **kwargs in target_function is passed here
        constraints=constraints,
        scaler=scaler,
        alpha=alpha,
        beta=beta,
        classifier=model_pipeline,
        x_clean=x_clean[j],
        y=y_clean[j],
        eps=eps,
        distance=distance,
        features_min_max=features_min_max,
        int_features=int_features,
        generate_perturbation=generate_perturbation,
        history=history,
        candidate=None
    )

    #print(len(trajectory), len(runtime), len(history), end="\n\n")

    # Last recorded function evaluation
    last_eval = history[-1]
    config, score, cost, budget, _info = last_eval
    conf_dict = dict(dehb.vector_to_configspace(config))
    l = list(conf_dict.keys())
    for i in range(len(l)):
        l[i] = l[i][7:]
    for i,k in zip(l, list(conf_dict.keys())):
        conf_dict[i] = conf_dict.pop(k)
    
    print(f'conf_dict {conf_dict}')
    s = sorted([int(i) for i in conf_dict.keys()])
    s = [str(i) for i in s]
    d = {}
    for k in s:
        d[k] = conf_dict[k]

    configuration = np.where(np.array(list(d.values())) == 1)[0]
    print(f'configuration {configuration}')
    '''
    print("Last evaluated configuration, ")
    print(dehb.vector_to_configspace(config), end="")
    print("got a score of {}, was evaluated at a budget of {:.2f} and "
        "took {:.3f} seconds to run.".format(score, budget, cost))
    print("The additional info attached: {}".format(_info))
    '''

    problem = AdversarialProblem(
        x_clean=x_clean[j], 
        n_var=len(configuration), 
        y_clean=y_clean[j], 
        classifier=model_pipeline, 
        constraints=constraints, 
        features_min_max=features_min_max, 
        scaler=scaler, 
        configuration=configuration, 
        int_features=int_features, 
        eps=0.2
    )
                
    ref_points = get_reference_directions(
        "energy", problem.n_obj, max_budget, seed=1
    )
    '''
                ref_points = get_reference_directions(
                        name="das-dennis", n_partitions=12, n_dim=problem.n_obj, n_points=91, seed=1
                )
    '''
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

    res = minimize(problem, algorithm, termination=('n_gen', 150))

    optimal_solutions = res.pop.get("X")
    optimal_objectives = res.pop.get("F")

    candidates.append(_info['adv'])
    optimals = []
    best_objectives = []
    for i in range(len(optimal_solutions)):
        '''
                    adv = np.copy(x)
                    adv[list(configuration)] += np.nan_to_num(optimal_solutions[i])
                    adv = np.clip(adv, features_min_max[0], features_min_max[1])
                    adv = self.fix_feature_types(optimal_solutions[i], adv, int_features, configuration)
                    pred = classifier.predict_proba(adv[np.newaxis, :])[0]
                    print(f'pred of the optimal solution {i} : {pred}')
        '''
                    #print(f"Objective values {i}: {optimal_objectives[i]}")
        if optimal_objectives[i][1] <= 0.2:
                        #print(f"Objective values {i}: {optimal_objectives[i]}")
            optimals.append(optimal_objectives[i])
                    #scores.append((sum(optimal_objectives[i]), optimal_solutions[i]))
    if len(optimals) > 0:
        optimals = sorted(optimals, key=lambda k: k[0])
        best_objectives.append(optimals[0])
        '''
                fronts = fast_non_dominated_sort.fast_non_dominated_sort(np.array(optimals))
                for front in fronts:
                    print("Front:")
                    for solution in front:
                        print(optimals[solution])
                    print("------------------")
                '''
    best_objectives = sorted(best_objectives, key=lambda k: k[0])
    print(f'best objectives for example {j} : {best_objectives}')
    if len(best_objectives) > 0:
        kk += 1
        final_objectives.append((best_objectives[0], j))
            
        
print(f'final objectives across all examples {final_objectives}')
sr = 0
cr = (model_pipeline.predict(x_clean[:BATCH_SIZE]) == 1).astype('int').sum()
for i, (obj, j) in enumerate(final_objectives):
    print(f'pred for example {j} : {model_pipeline.predict(x_clean[j][np.newaxis, :])[0]}')
    if model_pipeline.predict(x_clean[j][np.newaxis, :])[0] != 1:
        print(f'pred inside the if for example {j} {model_pipeline.predict(x_clean[j][np.newaxis, :])[0]}')
        continue

    if obj[0] < 0.5 and obj[2] <= 0.00001:
        print(f'adversarial {j} : pred {obj[0]}')
        sr += 1
print(f'Correct {cr}')
print(f'Success rate {(sr / (cr + kk)) * 100 }%')

end = timeit.default_timer()
print(f'It took {(end - start) / 60}')
candidates = np.array(candidates)
preds = model_pipeline.predict(candidates)
sr = (preds != 1).astype('int').sum() / BATCH_SIZE
print(sr)