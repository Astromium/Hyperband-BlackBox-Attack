from ConfigSpace import ConfigurationSpace
from numpy.random import choice
from scipy.special import softmax
from typing import List

class ConfigGenerator():
    def __init__(self, mutable_features: List, max_config_size: int) -> None:
        self.max_config_size = max_config_size
        self.mutable_features = mutable_features
        self.space = {f'{i}': (0.0, 0.1) for i in range(len(self.mutable_features))}
        self.space['size'] = (1, max_config_size)
        self.cs = ConfigurationSpace(space=self.space)
    
    def get_samples(self, n_sample: int):
        samples = self.cs.sample_configuration(n_sample)
        return [self.transform_sample(dict(sample)) for sample in samples] 
    
    def transform_sample(self, sample):
        s = [0] * (len(self.mutable_features) + 1)
        for k in list(sample.keys())[:-1]:
            s[int(k)] = sample[k]
        s[-1] = sample['size']
        return s
    
    def get_configurations(self, n_sample: int):
        samples = self.get_samples(n_sample=n_sample)
        return [choice(self.mutable_features, sample[-1], False, softmax(sample[:-1])) for sample in samples], samples
    
