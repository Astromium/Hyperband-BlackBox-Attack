import math
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, List, Union
from numpy.typing import NDArray
from sampler import Sampler
from evaluators import Evaluator
from utils.perturbation_generator import generate_perturbation
from bayes_opt import BayesianOptimizer
from utils.config_generator2 import ConfigGenerator

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
    optimizer: BayesianOptimizer
    config_generator: ConfigGenerator
    is_first: bool
    
    def run(self):
        if(self.downsample <= 1):
            raise(ValueError('Downsample must be > 1'))
        
        round_n = lambda n : max(round(n), 1)

        # Sample without prior if first bracket
        if self.is_first:
            configurations = self.config_generator.get_configurations(n_sample=self.n_configurations, logits=None)
        else:
            configuration = self.optimizer.get_next(n_samples=self.n_configurations)

        scores = [math.inf for _ in range(len(configurations))]
        candidates = [None for _ in range(len(configurations))]

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


    
