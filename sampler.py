import random
from dataclasses import dataclass
from typing import List, Union

@dataclass
class Sampler():

    def sample(self, dimensions: int, num_configs: int, max_configuration_size: int, mutables_mask: Union[List[int], None], seed: int) -> List[int]:
        #random.seed(seed)
        configurations = [None] * num_configs
        if mutables_mask:
            sample_list = mutables_mask
        else:
            sample_list = list(range(0, dimensions))
        
        for i in range(num_configs):
            n = random.randint(40, max_configuration_size + 1)
            config = random.sample(sample_list, n)
            configurations[i] = config

        return configurations