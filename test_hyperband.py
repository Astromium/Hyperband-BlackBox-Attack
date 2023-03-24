from evaluators import Evaluator
import numpy as np
from hyperband import Hyperband
from sampler import Sampler
from utils.perturbation_generator import generate_perturbation

class URLEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, classifier, configuration, budget, x, y, eps, distance, features_min_max, generate_perturbation):
        s = 0
        for i in range(budget):
            s += sum(list(configuration))
        return s, configuration
    
url_evaluator = URLEvaluator()
sampler = Sampler()
x = np.array([1,2,3])
y=1
    
hp = Hyperband(objective=url_evaluator, classifier=None, sampler=sampler, x=x, y=y, eps=0.1, dimensions=20, max_configuration_size=10, R=81, distance='l2', downsample=3)
all_scores, all_configs, all_candidates = hp.generate()

print(f'scores {all_scores}')
print(f'configs {all_configs}')
print(f'candidates {all_candidates}')