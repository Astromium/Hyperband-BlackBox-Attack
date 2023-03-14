import math
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, List, Union
from numpy.typing import NDArray
from sampler import Sampler
from evaluators import Evaluator
from utils.perturbation_generator import generate_perturbation

@dataclass
class SuccessiveHalving():
    objective: Evaluator
    classifier: Any
    sampler: Sampler 
    x: NDArray
    y: int 
    eps: float 
    dimensions: int 
    max_configuration_size: int
    distance: str
    max_ressources_per_configuration: int
    downsample: int
    bracket_budget: int
    n_configurations: int
    mutables: Union[List, None]
    features_min_max: Union[List, None]
    hyperband_bracket: int
    
    def run(self):
        if(self.downsample <= 1):
            raise(ValueError('Downsample must be > 1'))
        
        round_n = lambda n : max(round(n), 1)
        
        configurations = self.sampler.sample(
            dimensions=self.dimensions,
            num_configs=self.n_configurations,
            max_configuration_size=self.max_configuration_size,
            mutables_mask=self.mutables
        )

        scores = [math.inf for i in range(len(configurations))]
        candidates = [None for i in range(len(configurations))]

        results = []
        for i in range(self.hyperband_bracket + 1):
            budget = self.bracket_budget * pow(self.downsample, i)
            for score, candidate, configuration in tqdm(zip(scores, candidates, configurations), total=len(configurations), desc=f'Running Round {i} of SH. Evaluating {len(configurations)} configurations with budget of {budget}'):
                new_score, new_candidate = self.objective.evaluate(
                    classifier=self.classifier,
                    configuration=configuration,
                    budget=budget,
                    x=self.x,
                    y=self.y,
                    eps=self.eps,
                    distance=self.distance,
                    features_min_max=self.features_min_max,
                    generate_perturbation=generate_perturbation
                )
                if new_score < score:
                    results.append(tuple([new_score, new_candidate]))
                else:
                    results.append(tuple([score, candidate]))

            # Sort by minimum score
            results = sorted(zip(results, configurations), key=lambda k: k[0][0])
            # keep the best half
            results = results[:round_n(len(configurations) / self.downsample)]
            results, configurations = zip(*results)
            # both arrays get casted to tuples for some reason
            results, configurations = list(results), list(configurations)
        scores, candidates = zip(*results)
        scores, candidates = list(scores), list(candidates)

        return scores, configurations, candidates


    
