import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mlc.datasets.dataset_factory import get_dataset
from hyperband import Hyperband
from evaluators import TfEvaluator
from sampler import Sampler
from utils.sr_calculators import TfCalculator
import timeit

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

#load model
model = tf.keras.models.load_model('./ressources/model_botnet.h5')

#get mutable features
mutables = metadata.index[metadata['mutable'] == True].tolist()

#parameters for Hyperband
BATCH_SIZE = 200 #X_test_botnet.shape[0]
perturbations = [0.05]
distance = 'inf'
dimensions = X_test.shape[1]
sampler = Sampler()
features_min_max = (0, 1)
botnet_evaluator = TfEvaluator()

success_rates_l2 = []
exec_times_l2 = []


if __name__ == '__main__':

    for eps in perturbations:
        start = timeit.default_timer()
        scores, configs, candidates = [], [], []

        for i in range(BATCH_SIZE):
            hp = Hyperband(objective=botnet_evaluator, classifier=model, x=X_test_botnet[i], y=y_test_botnet[i], sampler=sampler, eps=eps, dimensions=dimensions, max_configuration_size=len(mutables)-1, downsample=3, distance=distance)
            all_scrores, all_configs, all_candidates = hp.generate(mutables=mutables, features_min_max=features_min_max)

            scores.append(all_scrores)
            configs.append(all_configs)
            candidates.append(all_candidates)
        
        end = timeit.default_timer()
        success_rate_calculator = TfCalculator(classifier=model, data=X_test_botnet[:BATCH_SIZE], labels=y_test_botnet[:BATCH_SIZE], scores=np.array(scores), candidates=candidates)
        success_rate, adversarials = success_rate_calculator.evaluate()
        success_rates_l2.append(success_rate)
        exec_times_l2.append(round((end - start) / 60, 3))

        #print(f'\n Execution time {round((end - start) / 60, 3)} \n')
        #print(f'\n Success rate over {BATCH_SIZE} examples : {success_rate * 100}')
    
    print(f'\n Execution times {exec_times_l2} \n')
    print(f'\n Success rates over {BATCH_SIZE} examples : {success_rates_l2}')
    dist = np.linalg.norm(adversarials[0][0] - X_test_botnet[0], ord=np.inf)
    print(f'dist {dist}')
