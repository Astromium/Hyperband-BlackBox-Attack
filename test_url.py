import tensorflow as tf
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
tf.compat.v1.disable_eager_execution()

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

model = tf.keras.models.load_model('./ressources/model_url.h5')

phishing = np.where(y_test == 1)[0]
X_test_phishing, y_test_phishing = X_test[phishing], y_test[phishing]

x_clean = np.load('./ressources/baseline_X_test_candidates.npy')
y_clean = np.load('./ressources/baseline_y_test_candidates.npy')
x_clean = scaler.transform(x_clean)

rf = joblib.load('./ressources/baseline_rf.model')
model_pipeline = Pipeline(
    steps=[('preprocessing', preprocessing_pipeline), ('model', rf)]
)
metadata = pd.read_csv('./ressources/url_metadata.csv')
min_constraints = metadata['min'].to_list()[:63]
max_constraints = metadata['max'].to_list()[:63]
features_min_max = (min_constraints, max_constraints)

constraints = get_url_relation_constraints()
executor = NumpyConstraintsExecutor(AndConstraint(constraints))

# Parameters for Hyperband
dimensions = X_test.shape[1]
BATCH_SIZE = 200
eps = 0.3
url_evaluator = TfEvaluator(constraints=constraints, scaler=scaler)
sampler = Sampler()
distance = 'inf'
success_rates_l2 = []
exec_times_l2 = []



if __name__ == '__main__':
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
    '''
    preds = model.predict(x_clean[:BATCH_SIZE])
    from scipy.special import softmax
    preds = np.argmax(softmax(preds, axis=1), axis=1)
    args_correct = (preds == y_clean[:BATCH_SIZE]).astype('int')
    x_correct, y_correct = x_clean[args_correct], y_clean[args_correct]
        
    
    scores, configs, candidates = [], [], []
    start = timeit.default_timer()
    for i in range(BATCH_SIZE):
        hp = Hyperband(objective=url_evaluator, classifier=model, x=x_clean[i], y=y_clean[i], sampler=sampler, eps=eps, dimensions=dimensions, max_configuration_size=dimensions-1, R=81, downsample=3, distance=distance)
        all_scrores, all_configs, all_candidates = hp.generate(mutables=None, features_min_max=(0,1))

        scores.append(all_scrores)
        configs.append(all_configs)
        candidates.append(all_candidates)

    end = timeit.default_timer()
    success_rate_calculator = TfCalculator(classifier=model, data=x_clean[:BATCH_SIZE], labels=y_clean[:BATCH_SIZE], scores=np.array(scores), candidates=candidates)
    success_rate, adversarials = success_rate_calculator.evaluate()
    adversarials = scaler.inverse_transform(np.array(adversarials))
    print(f'\n Execution Time {round((end - start) / 60, 3)}\n')
    print(f'Success rate over {BATCH_SIZE} examples (M) : {success_rate * 100}')
    print(f'len adversarials {len(adversarials)}')
    violations = np.array([executor.execute(adv[np.newaxis, :])[0] for adv in adversarials])
    tolerance = 0.0001
    satisfaction = round(((violations > tolerance).astype('int').sum() / len(adversarials)) * 100, 3)
    print(f'Constraints satisfaction (C&M) {(success_rate * 100) - satisfaction}')
    
    #scores = softmax(model.predict(np.array(adversarials)), axis=1)
    #print(f'scores {scores}')
    #print(f'Violations for x_clean {[executor.execute(x[np.newaxis, :]) for x in x_clean]}')
    #dist = np.linalg.norm(adversarials[0][0] - X_test_phishing[0])
    #print(f'dist {dist}')
    
