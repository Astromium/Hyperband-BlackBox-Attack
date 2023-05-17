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
import joblib


seed = 202374
np.random.seed(seed)
warnings.filterwarnings('ignore')
history = {}

min_budget, max_budget = 2, 81

import ConfigSpace as CS


def create_search_space(seed=202374):
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
BATCH_SIZE = x_clean.shape[0]
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
n_iterations = 5


dehb = DEHB(
    f=target_function, 
    cs=cs, 
    dimensions=dimensions, 
    min_budget=min_budget, 
    max_budget=max_budget,
    n_workers=1,
    output_path="./temp"
)

alpha=1.0
beta=1.0
candidates = []
for i in range(BATCH_SIZE):
    dehb.reset()
    trajectory, runtime, history = dehb.run(
        #total_cost=10,
        brackets=2,
        verbose=False,
        save_intermediate=False,
        # parameters expected as **kwargs in target_function is passed here
        constraints=constraints,
        scaler=scaler,
        alpha=alpha,
        beta=beta,
        classifier=model_pipeline,
        x_clean=x_clean[i],
        y=y_clean[i],
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
    '''
    print("Last evaluated configuration, ")
    print(dehb.vector_to_configspace(config), end="")
    print("got a score of {}, was evaluated at a budget of {:.2f} and "
        "took {:.3f} seconds to run.".format(score, budget, cost))
    print("The additional info attached: {}".format(_info))
    '''
    candidates.append(_info['adv'])

candidates = np.array(candidates)
preds = model_pipeline.predict(candidates)
sr = (preds != 1).astype('int').sum() / BATCH_SIZE
print(sr)