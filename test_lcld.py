import os
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
from constraints.relation_constraint import AndConstraint
from constraints.lcld_constraints import get_relation_constraints
from ml_wrappers import wrap_model
from utils.tensorflow_classifier import TensorflowClassifier
from tensorflow.keras.models import load_model
import timeit
import joblib
import pickle
import warnings
import cProfile, pstats
from sklearn.pipeline import Pipeline
from itertools import product
from mlc.datasets.dataset_factory import get_dataset
import tensorflow as tf
warnings.filterwarnings(action='ignore')
tf.compat.v1.disable_eager_execution()
scaler = preprocessing_pipeline = joblib.load('./ressources/custom_lcld_scaler.joblib')


if __name__ == '__main__':
    print(f'Hello {os.getpid()} from test_url')

    def process_encoded_min_max(mins, maxs, types):
        min_features = []
        max_features = []
        for i in range(len(types)):
            if types[i] != 'cat':
                min_features.append(mins[i])
                max_features.append(maxs[i])
            else:
                for j in range(int(mins[i]), int(maxs[i]) + 1):
                    min_features.append(0)
                    max_features.append(1)
        print(f'len min features {len(min_features)}')
        print(f'len max features {len(max_features)}')
        assert len(min_features) == 50
        assert len(max_features) == 50

        return min_features, max_features
    
    def process_encoded_mutables(mins, maxs, types, mutables):
        encoded_mutables = []
        for i in range(len(mutables)):
            if mutables[i] == True:
                if types[i] != 'cat':
                    encoded_mutables.append(i)
                else:
                    for j in range(mins[i], maxs[i] + 1):
                        pass 


    
    ds = get_dataset('lcld_v2_iid')
    splits = ds.get_splits()
    x, y = ds.get_x_y()
    feature_names = x.columns.to_list()
    categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']
    encoded_df = pd.get_dummies(x, columns=categorical)
    encoded_df = encoded_df.to_numpy()
    x_test, y_test = encoded_df[splits['test']], y[splits['test']]
    charged_off = np.where(y_test == 1)[0]
    x_charged_off, y_charged_off = x_test[charged_off], y_test[charged_off] 
    #x_clean = scaler.transform(x_clean)

    #model_pipeline = Pipeline(steps=[('preprocessing', preprocessing_pipeline), ('model', rf)])
    metadata = ds.get_metadata(only_x=True)
    min_constraints = metadata['min'].to_list()
    min_constraints = list(map(float, min_constraints))
    max_constraints = metadata['max'].to_list()
    max_constraints = list(map(float, max_constraints))
    feature_types = metadata['type'].to_list()
    min_constraints, max_constraints = process_encoded_min_max(min_constraints, max_constraints, feature_types)
    mutables = metadata.index[metadata['mutable'] == True].tolist()
    print(f'mutables {mutables}')
    int_features = np.where(np.array(feature_types) == 'int')[0]
    cat_features = np.where(np.array(feature_types) == 'cat')[0]
    int_features = list(int_features) + list(cat_features)
    features_min_max = (min_constraints, max_constraints)

    mutables = [0, 3, 5, 7, 10 ,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    constraints = get_relation_constraints()
    print(f'constraints {constraints}')
    executor = NumpyConstraintsExecutor(AndConstraint(constraints), feature_names=feature_names)
    # Parameters for Hyperband
    dimensions = x_charged_off.shape[1] if mutables is None else len(mutables)
    eps = 0.2
    downsample = 3
    sampler = Sampler()
    distance = 2
    classifier_path = './ressources/custom_lcld_model.h5'
    model = tf.keras.models.load_model(classifier_path)
    model_pipeline = Pipeline(steps=[('preprocessing', preprocessing_pipeline), ('model', model)])
    preds = model_pipeline.predict(x_charged_off)
    classes = np.argmax(preds, axis=1)
    print(f'preds {classes}')
    to_keep = np.where(classes == 1)[0]
    print(f'Correct {to_keep.size}')
    x_charged_off_correct, y_charged_off_correct = x_charged_off[to_keep], y_charged_off[to_keep]
    print(f'shape of test set {x_charged_off_correct.shape}') 
    BATCH_SIZE = x_charged_off_correct.shape[0]
    #print(model.summary())
    seed = 202374
    #np.random.seed(seed)
    success_rates_l2 = []
    exec_times_l2 = []

    R_values = [81]
    epsilons = [0.2]
    params = list(product(R_values, epsilons))
    print(params)
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
        
    for (R, eps) in params:
        url_evaluator = TorchEvaluator(constraints=None, scaler=scaler, alpha=1.0, beta=1.0, feature_names=feature_names)
        scores, configs, candidates = [], [], []
        start = timeit.default_timer()
        
        hp = Hyperband(objective=url_evaluator, classifier_path=classifier_path, x=x_charged_off_correct[:BATCH_SIZE], y=y_charged_off_correct[:BATCH_SIZE], sampler=sampler, eps=eps, dimensions=dimensions, max_configuration_size=dimensions-1, R=R, downsample=downsample, distance=distance, seed=seed)
        profiler = cProfile.Profile()
        profiler.enable()
        scores, configs, candidates, _, _, _ = hp.generate(mutables=mutables, features_min_max=(min_constraints,max_constraints), int_features=int_features)
        profiler.disable()
        #print(f'scores {scores}')
        #stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
        #stats.print_stats()
        #stats.dump_stats('results.prof')

        end = timeit.default_timer()
        print(f'Exec time {round((end - start) / 60, 3)}')
        #model_tf = TensorflowClassifier(load_model(classifier_path))
        model_tf = tf.keras.models.load_model(classifier_path)
        model_pipeline = Pipeline(steps=[('preprocessing', preprocessing_pipeline), ('model', wrap_model(model_tf, x_charged_off_correct[:BATCH_SIZE], model_task='classification'))])
        success_rate_calculator = TorchCalculator(classifier=model_pipeline, data=x_charged_off_correct[:BATCH_SIZE], labels=y_charged_off_correct[:BATCH_SIZE], scores=np.array(scores), candidates=candidates, scaler=scaler, eps=eps)
        success_rate, best_candidates, adversarials = success_rate_calculator.evaluate()
        print(f'success rate {success_rate}, len best_candidates {len(best_candidates)}, len adversarials {len(adversarials)}')
        #adversarials, best_candidates = scaler.inverse_transform(np.array(adversarials)), scaler.inverse_transform(np.array(best_candidates))
        np.save('./adversarials_lcld.npy', np.array(adversarials))
        violations = np.array([executor.execute(adv[np.newaxis, :])[0] for adv in adversarials])
        violations_candidates = np.array([executor.execute(adv[np.newaxis, :])[0] for adv in best_candidates])
        tolerance = 0.0001
        satisfaction = (violations < tolerance).astype('int').sum()
        satisfaction_candidates = (violations_candidates < tolerance).astype('int').sum()
        #print(f'Constraints satisfaction (C&M) {(success_rate * 100) - satisfaction}')
        history_dict[(R, eps)] = {'M': round(success_rate * 100, 2), 'C&M': round((satisfaction * 100) / BATCH_SIZE, 2), 'C': round((satisfaction_candidates * 100) / len(best_candidates), 2), 'Execution time': round((end - start) / 60, 3)}
        
    
    print(f'History {history_dict}')
    with open('experiments.pkl', 'wb') as f:
        pickle.dump(history_dict, f)
    
    #scores = softmax(model.predict(np.array(adversarials)), axis=1)
    #print(f'scores {scores}')
    #print(f'Violations for x_clean {[executor.execute(x[np.newaxis, :]) for x in x_clean]}')
    #dist = np.linalg.norm(adversarials[0][0] - X_test_phishing[0])
    #print(f'dist {dist}')
    
