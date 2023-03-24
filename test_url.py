import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_v2_behavior()
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from hyperband import Hyperband
from evaluators import TfEvaluator
from sampler import Sampler
from utils.sr_calculators import TfCalculator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.url_constraints import get_url_relation_constraints
from constraints.relation_constraint import AndConstraint
from sklearn.pipeline import Pipeline
import timeit
import joblib
import pickle
import warnings
#tf.compat.v1.enable_v2_behavior()
warnings.filterwarnings(action='ignore')

load_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

model = tf.keras.models.load_model('./ressources/model_url2.h5', options=load_option)


if __name__ == '__main__':
    df = pd.read_csv('./ressources/url.csv')
    X = np.array(df[df.columns.drop('is_phishing')])
    y = np.array(df['is_phishing'])

    merged = np.arange(len(X))
    i_train, i_test = train_test_split(
        merged,
        random_state=100,
        shuffle=True,
        stratify=y[merged],
        test_size=0.2
    )

    #scaler = MinMaxScaler()
    scaler = preprocessing_pipeline = joblib.load('./ressources/baseline_scaler.joblib')
    X = scaler.transform(X)
    X_train, X_test = X[i_train], X[i_test]
    y_train, y_test = y[i_train], y[i_test]

    

    phishing = np.where(y_test == 1)[0]
    X_test_phishing, y_test_phishing = X_test[phishing], y_test[phishing]

    x_clean = np.load('./ressources/baseline_X_test_candidates.npy')
    y_clean = np.load('./ressources/baseline_y_test_candidates.npy')
    x_clean = scaler.transform(x_clean)

    rf = joblib.load('./ressources/baseline_rf.model')
    #model_pipeline = Pipeline(steps=[('preprocessing', preprocessing_pipeline), ('model', rf)])
    metadata = pd.read_csv('./ressources/url_metadata.csv')
    min_constraints = metadata['min'].to_list()[:63]
    max_constraints = metadata['max'].to_list()[:63]
    features_min_max = (min_constraints, max_constraints)

    constraints = get_url_relation_constraints()
    executor = NumpyConstraintsExecutor(AndConstraint(constraints))

    # Parameters for Hyperband
    dimensions = X_test.shape[1]
    BATCH_SIZE = 10
    eps = 0.2
    sampler = Sampler()
    distance = 'l2'
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
        url_evaluator = TfEvaluator(constraints=constraints, scaler=scaler, alpha=0.5, beta=0.5)
        scores, configs, candidates = [], [], []
        start = timeit.default_timer()
        for i in range(BATCH_SIZE):
            hp = Hyperband(objective=url_evaluator, classifier=model, x=x_clean[i], y=y_clean[i], sampler=sampler, eps=eps, dimensions=dimensions, max_configuration_size=dimensions-1, R=R, downsample=3, distance=distance)
            all_scrores, all_configs, all_candidates = hp.generate(mutables=None, features_min_max=(0,1))

            scores.append(all_scrores)
            configs.append(all_configs)
            candidates.append(all_candidates)
            print(f'all_scores length {all_scrores}')

        end = timeit.default_timer()
        print(f'Exec time {round((end - start) / 60, 3)}')
        success_rate_calculator = TfCalculator(classifier=model, data=x_clean[:BATCH_SIZE], labels=y_clean[:BATCH_SIZE], scores=np.array(scores), candidates=candidates)
        success_rate, best_candidates, adversarials = success_rate_calculator.evaluate()
        print(f'success rate {success_rate}, len best_candidates {len(best_candidates)}, len adversarials {len(adversarials)}')
        adversarials, best_candidates = scaler.inverse_transform(np.array(adversarials)), scaler.inverse_transform(np.array(best_candidates))
        #print(f'\n Execution Time {round((end - start) / 60, 3)}\n')
        #print(f'Success rate over {BATCH_SIZE} examples (M) : {success_rate * 100}')
        #print(f'len adversarials {len(adversarials)}')
        violations = np.array([executor.execute(adv[np.newaxis, :])[0] for adv in adversarials])
        violations_candidates = np.array([executor.execute(adv[np.newaxis, :])[0] for adv in best_candidates])
        tolerance = 0.0001
        satisfaction = (violations < tolerance).astype('int').sum()
        satisfaction_candidates = (violations_candidates < tolerance).astype('int').sum()
        #print(f'Constraints satisfaction (C&M) {(success_rate * 100) - satisfaction}')
        history_dict[R] = {'M': round(success_rate * 100, 2), 'C&M': round((satisfaction * 100) / BATCH_SIZE, 2), 'C': round((satisfaction_candidates * 100) / len(best_candidates), 2), 'Execution time': round((end - start) / 60, 3)}
    
    print(f'History {history_dict}')
    with open('history.pkl', 'wb') as f:
        pickle.dump(history_dict, f)
    
    #scores = softmax(model.predict(np.array(adversarials)), axis=1)
    #print(f'scores {scores}')
    #print(f'Violations for x_clean {[executor.execute(x[np.newaxis, :]) for x in x_clean]}')
    #dist = np.linalg.norm(adversarials[0][0] - X_test_phishing[0])
    #print(f'dist {dist}')
    
