from ConfigSpace import ConfigurationSpace
from numpy.random import choice
from scipy.special import softmax
from typing import List, Union
from skopt import Space
from skopt.space import Real, Integer

class ConfigGenerator():
    def __init__(self, mutable_features: List) -> None:
        self.mutable_features = mutable_features
        self.space = [Real(0, 1) for i in range(len(self.mutable_features))]
        self.space.append(Integer(1, len(self.mutable_features)))
        self.cs = Space(dimensions=self.space)
    
    def get_samples(self, n_sample: int):
        return self.cs.rvs(n_samples=n_sample) 
    
    def get_configurations(self, n_sample: int, logits: Union[List, None]):
        if logits is not None:
            samples = logits
        else:
            samples = self.get_samples(n_sample=n_sample)
        out = list(zip([choice(self.mutable_features, int(sample[-1]), False, softmax(sample[:-1])) for sample in samples], samples))
        return out
        
    
