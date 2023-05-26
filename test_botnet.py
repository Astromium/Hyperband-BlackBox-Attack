import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
import pandas as pd
from hyperband import Hyperband
from evaluators import TorchEvaluator
from sampler import Sampler
from utils.sr_calculators import TorchCalculator
from sklearn.model_selection import train_test_split
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.url_constraints import get_url_relation_constraints
from constraints.relation_constraint import AndConstraint
from ml_wrappers import wrap_model
from utils.model import Net
from utils.tensorflow_classifier import TensorflowClassifier
from tensorflow.keras.models import load_model
import timeit
import joblib
import pickle
import warnings
import cProfile, pstats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from mlc.datasets.dataset_factory import get_dataset
warnings.filterwarnings(action='ignore')

# load dataset
ds = get_dataset('ctu_13_neris')
X, y = ds.get_x_y()
metadata = ds.get_metadata()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test, y_test = X[143046:], y[143046:]

# filter non botnet examples
botnet = np.where(y_test == 1)[0]
X_test_botnet, y_test_botnet = X_test[botnet], y_test[botnet]

scaler = preprocessing_pipeline = joblib.load('./ressources/baseline_scaler.joblib')



#get mutable features
mutables = metadata.index[metadata['mutable'] == True].tolist()
min_constraints = metadata['min'].to_list()[:-1]
max_constraints = metadata['max'].to_list()[:-1]
features_min_max = (min_constraints, max_constraints)
feature_types = metadata.index[metadata['type'] == 'int'].to_list()
#print(f'metadata {metadata}')


if __name__ == '__main__':
    print(f'Hello {os.getpid()} from test_url')
    constraints = get_url_relation_constraints()
    executor = NumpyConstraintsExecutor(AndConstraint(constraints))

    model_pipeline = Pipeline(steps=[('preprocessing', scaler), ('model', model)])

    # Parameters for Hyperband
    dimensions = X_test.shape[1]
    BATCH_SIZE = 500#x_clean.shape[0]
    eps = 0.2
    downsample = 3
    sampler = Sampler()
    distance = 2
    classifier_path = './ressources/model_botnet.h5'
    seed = 202374
    np.random.seed(seed)
    success_rates_l2 = []
    exec_times_l2 = []

    R_values = [81]
    history_dict = dict()
    '''
    for eps in perturbations:
        start = timeit.default_timer()
        scores, configs, candidates = [], [], []

        for i in range(BATCH_SIZE):
            hp = Hyperband(objective=url_evaluator, classifier=model, x=X_test_phishing[i], y=y_test_phishing[i], sampler=sampler, eps=eps, dimensions=dimensions, max_configuration_size=dimensions-1, downsample=3, distance=distance)
            all_scrores, all_configs, all_candidates = hp.generate(mutables=None, features_min_max=(0,1))

            scores.append(all_scrores)
            configs.append(all_configs)
            candidates.append(all_candidates)
        
        end = timeit.default_timer()
        success_rate_calculator = TfCalculator(classifier=model, data=X_test_phishing[:BATCH_SIZE], labels=y_test_phishing[:BATCH_SIZE], scores=np.array(scores), candidates=candidates)
        success_rate, adversarials = success_rate_calculator.evaluate()
        success_rates_l2.append(success_rate)
        exec_times_l2.append(round((end - start) / 60, 3))

    print(f'\n Execution Time {exec_times_l2}\n')
    print(f'Success rate over {BATCH_SIZE} examples : {success_rates_l2}')
    print(f'len adversarials {len(adversarials)}')
    preds = model.predict(x_clean[:BATCH_SIZE])
    from scipy.special import softmax
    preds = np.argmax(softmax(preds, axis=1), axis=1)
    args_correct = (preds == y_clean[:BATCH_SIZE]).astype('int')
    x_correct, y_correct = x_clean[args_correct], y_clean[args_correct]
    '''
        
    for R in R_values:
        url_evaluator = TorchEvaluator(constraints=None, scaler=scaler, alpha=1.0, beta=1.0)
        scores, configs, candidates = [], [], []
        start = timeit.default_timer()
        
        hp = Hyperband(objective=url_evaluator, classifier_path=classifier_path, x=x_clean[:BATCH_SIZE], y=y_clean[:BATCH_SIZE], sampler=sampler, eps=eps, dimensions=dimensions, max_configuration_size=dimensions-1, R=R, downsample=downsample, distance=distance, seed=seed)
        profiler = cProfile.Profile()
        profiler.enable()
        scores, configs, candidates, _, _, _ = hp.generate(mutables=None, features_min_max=(min_constraints,max_constraints), int_features=int_features)
        profiler.disable()
        #stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
        #stats.print_stats()
        #stats.dump_stats('results.prof')

        end = timeit.default_timer()
        print(f'Exec time {round((end - start) / 60, 3)}')
        #print(f'scores {scores}')
        #load model
        model = tf.keras.models.load_model('./ressources/model_botnet.h5')
        model_pipeline = Pipeline(steps=[('preprocessing', scaler), ('model', model)])
        success_rate_calculator = TorchCalculator(classifier=model_pipeline, data=x_clean[:BATCH_SIZE], labels=y_clean[:BATCH_SIZE], scores=np.array(scores), candidates=candidates, scaler=scaler)
        success_rate, best_candidates, adversarials = success_rate_calculator.evaluate()
        print(f'success rate {success_rate}, len best_candidates {len(best_candidates)}, len adversarials {len(adversarials)}')
        #adversarials, best_candidates = scaler.inverse_transform(np.array(adversarials)), scaler.inverse_transform(np.array(best_candidates))
        '''
        violations = np.array([executor.execute(adv[np.newaxis, :])[0] for adv in adversarials])
        violations_candidates = np.array([executor.execute(adv[np.newaxis, :])[0] for adv in best_candidates])
        tolerance = 0.0001
        satisfaction = (violations < tolerance).astype('int').sum()
        satisfaction_candidates = (violations_candidates < tolerance).astype('int').sum()
        #print(f'Constraints satisfaction (C&M) {(success_rate * 100) - satisfaction}')
        history_dict[R] = {'M': round(success_rate * 100, 2), 'C&M': round((satisfaction * 100) / BATCH_SIZE, 2), 'C': round((satisfaction_candidates * 100) / len(best_candidates), 2), 'Execution time': round((end - start) / 60, 3)}
    
    print(f'History {history_dict}')
    '''
    
    
    #scores = softmax(model.predict(np.array(adversarials)), axis=1)
    #print(f'scores {scores}')
    #print(f'Violations for x_clean {[executor.execute(x[np.newaxis, :]) for x in x_clean]}')
    #dist = np.linalg.norm(adversarials[0][0] - X_test_phishing[0])
    #print(f'dist {dist}')
    
