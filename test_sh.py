from evaluators import Evaluator
import numpy as np
from succesive_halving import SuccessiveHalving
from sampler import Sampler

class URLEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, classifier, configuration, budget, x, y, eps, distance, features_min_max):
        s = 0
        for i in range(budget):
            s += sum(list(configuration))
        return s, configuration
    
url_evaluator = URLEvaluator()
sampler = Sampler()
x = np.array([1,2,3])
y=1
    
sh = SuccessiveHalving(objective=url_evaluator, clf=None, sampler=sampler, x=x, y=y, eps=0.1, dimensions=5, max_configuration_size=4, distance='l2', hyperband_bracket=4)
scores, configs, candidates = sh.run()

print(f'scores {scores}')
print(f'configs {configs}')
print(f'candidates {candidates}')