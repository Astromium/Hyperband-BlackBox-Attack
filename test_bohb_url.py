import pstats
import cProfile
import warnings
import pickle
import joblib
import timeit
from tensorflow.keras.models import load_model
from utils.model import Net
from ml_wrappers import wrap_model
from constraints.relation_constraint import AndConstraint
from constraints.url_constraints import get_url_relation_constraints
from constraints.constraints_executor import NumpyConstraintsExecutor
from sklearn.pipeline import Pipeline
from utils.sr_calculators import TorchCalculator
from utils.tensorflow_classifier import TensorflowClassifier
from sampler import Sampler
from evaluators import TorchEvaluator
from main_worker import MainWorker as worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB
from utils.perturbation_generator import generate_perturbation
import pandas as pd
import numpy as np
import torch
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore')

scaler = preprocessing_pipeline = joblib.load(
    './ressources/baseline_scaler.joblib')

logger = logging.getLogger()
logger.disabled = True

files_to_delete = ['results.json', 'configs.json', 'results.pkl']
for fname in files_to_delete:
    if os.path.exists(fname):
        os.remove(fname)

if __name__ == '__main__':

    x_clean = np.load('./ressources/baseline_X_test_candidates.npy')
    y_clean = np.load('./ressources/baseline_y_test_candidates.npy')
    # x_clean = scaler.transform(x_clean)

    metadata = pd.read_csv('./ressources/url_metadata.csv')
    min_constraints = metadata['min'].to_list()[:63]
    max_constraints = metadata['max'].to_list()[:63]
    feature_types = metadata['type'].to_list()[:63]
    int_features = np.where(np.array(feature_types) == 'int')[0]
    features_min_max = (min_constraints, max_constraints)

    constraints = get_url_relation_constraints()
    executor = NumpyConstraintsExecutor(AndConstraint(constraints))

    model_tf = TensorflowClassifier(
        load_model(r'ressources\baseline_nn.model'))
    model = Net()
    model = torch.load('./ressources/model_url.pth')
    model = wrap_model(model, x_clean, model_task='classification')
    model_pipeline = Pipeline(
        steps=[('preprocessing', preprocessing_pipeline), ('model', model_tf)])

    # Parameters for Hyperband
    dimensions = x_clean.shape[1]
    BATCH_SIZE = 10  # x_clean.shape[0]
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

    result_logger = hpres.json_result_logger(
        directory=shared_directory, overwrite=False)


# Start local worker
# w = worker(constraints=constraints, scaler=scaler, alpha=1.0, beta=1.0, classifier=model_pipeline, x=x_clean[0].tolist(), y=y_clean[0], eps=eps, distance=distance, features_min_max=features_min_max, int_features=int_features, generate_perturbation=generate_perturbation, history={}, candidate=None, run_id=run_id, host=host, nameserver=ns_host,
#            nameserver_port=ns_port, timeout=120)
# w.run(background=True)

workers = []
candidates = []
start = timeit.default_timer()
for i in range(BATCH_SIZE):
    # Start a nameserver:
    NS = hpns.NameServer(run_id=run_id, host=host, port=0,
                         working_directory=shared_directory)
    ns_host, ns_port = NS.start()
    w = worker(constraints=constraints, scaler=scaler, alpha=1.0, beta=1.0, classifier=model_pipeline, x=x_clean[i].tolist(), y=y_clean[i], eps=eps, distance=distance, features_min_max=features_min_max, int_features=int_features, generate_perturbation=generate_perturbation, history={}, candidate=None, run_id=run_id, host=host, nameserver=ns_host,
               nameserver_port=ns_port, timeout=120)
    w.run(background=True)
    # Run an optimizer
    bohb = BOHB(configspace=worker.get_configspace(),
                run_id=run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                result_logger=result_logger,
                min_budget=min_budget, max_budget=max_budget,
                )
    results = bohb.run(n_iterations=n_iterations)
    id2config = results.get_id2config_mapping()
    incumbent = results.get_incumbent_id()
    incumbent_config = id2config[incumbent]['config']
    run = results.get_runs_by_id(incumbent)
    cd = np.array(run[-1].__getitem__("info")["candidate"])
    candidates.append(cd)
    # shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


end = timeit.default_timer()
print(f'exec time {(end - start) / 60}')
success_rate_calculator = TorchCalculator(
    classifier=model_pipeline, data=x_clean[:BATCH_SIZE], labels=y_clean[:BATCH_SIZE], candidates=candidates, scaler=scaler)
sr = success_rate_calculator.evaluate()
print(f'sr {sr}')


# url_evaluator = TorchEvaluator(
#     constraints=constraints, scaler=scaler, alpha=1.0, beta=1.0)
# scores, configs, candidates = [], [], []
# start = timeit.default_timer()

# end = timeit.default_timer()
# print(f'Exec time {round((end - start) / 60, 3)}')
# model_nn = TensorflowClassifier(
#     load_model('./ressources/baseline_nn.model'))
# model_pipeline = Pipeline(
#     steps=[('preprocessing', preprocessing_pipeline), ('model', model_nn)])
# success_rate_calculator = TorchCalculator(
#     classifier=model_pipeline, data=x_clean[:BATCH_SIZE], labels=y_clean[:BATCH_SIZE], scores=np.array(scores), candidates=candidates, scaler=scaler)
# success_rate, best_candidates, adversarials = success_rate_calculator.evaluate()
# print(
#     f'success rate {success_rate}, len best_candidates {len(best_candidates)}, len adversarials {len(adversarials)}')
# adversarials, best_candidates = scaler.inverse_transform(
#     np.array(adversarials)), scaler.inverse_transform(np.array(best_candidates))

# violations = np.array(
#     [executor.execute(adv[np.newaxis, :])[0] for adv in adversarials])
# violations_candidates = np.array(
#     [executor.execute(adv[np.newaxis, :])[0] for adv in best_candidates])
# tolerance = 0.0001
# satisfaction = (violations < tolerance).astype('int').sum()
# satisfaction_candidates = (
#     violations_candidates < tolerance).astype('int').sum()
# # print(f'Constraints satisfaction (C&M) {(success_rate * 100) - satisfaction}')
# history_dict[R] = {'M': round(success_rate * 100, 2), 'C&M': round((satisfaction * 100) / BATCH_SIZE, 2), 'C': round(
#     (satisfaction_candidates * 100) / len(best_candidates), 2), 'Execution time': round((end - start) / 60, 3)}

# print(f'History {history_dict}')
