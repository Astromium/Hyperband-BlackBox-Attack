import math
import numpy as np
from tqdm import tqdm
from keras.models import load_model
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
    y: NDArray 
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
    int_features: Union[NDArray, None]
    seed: int
    hyperband_bracket: int

    def process_one(self, candidate, idx, configuration, budget, history):
        new_score, new_candidate = self.objective.evaluate(
           classifier=self.classifier,
            configuration=configuration,
            budget=budget,
            x=self.x[idx],
            y=self.y[idx],
            eps=self.eps,
            distance=self.distance,
            features_min_max=self.features_min_max,
            int_features=self.int_features,
            generate_perturbation=generate_perturbation,
            history=history,
            candidate=candidate 
        )
        history[tuple(configuration)].append(new_score)
        return new_score, new_candidate
        #if new_score < score:
            #return tuple([new_score, new_candidate])
        #else:
            #return tuple([score, candidate])
        
    def run_one(self, idx):

        configurations = self.sampler.sample(
                dimensions=self.dimensions,
                num_configs=self.n_configurations,
                max_configuration_size=self.max_configuration_size,
                mutables_mask=self.mutables,
                seed=self.seed
            )  
        history = {}
        #scores = [math.inf for s in range(len(configurations))]
        #candidates = [None for c in range(len(configurations))]

        for i in range(self.hyperband_bracket + 1):
            budget = self.bracket_budget * pow(self.downsample, i)
            results = [self.process_one(score=score, candidate=candidate, idx=idx, configuration=configuration, budget=budget, history=history) for score, candidate, configuration in zip(scores, candidates, configurations)]
            scores = [r[0] for r in results]
            candidates = [r[1] for r in results]
            top_indices = np.argsort(scores)[:max(int(len(scores) / self.downsample), 1)]
            configurations = [configurations[j] for j in top_indices]
            candidates = [candidates[j] for j in top_indices]
            '''
            results = sorted(zip(results, configurations), key=lambda k: k[0][0])
            # keep the best half
            results = results[:max(int(len(configurations) / self.downsample), 1)]
            results, configurations = zip(*results)
            # both arrays get casted to tuples for some reason
            results, configurations = list(results), list(configurations)
        
        scores, candidates = zip(*results)
        scores, candidates = list(scores), list(candidates)
        '''

        return (scores, configurations, candidates, history)



        
    
    def run(self):
        if(self.downsample <= 1):
            raise(ValueError('Downsample must be > 1'))
        
        
        round_n = lambda n : max(int(n), 1)
        #all_results = [self.run_one(idx=i) for i in range(self.x.shape[0])]
        #return all_results

        
        all_results = [] 

        for idx in range(self.x.shape[0]):
        
            configurations = self.sampler.sample(
                dimensions=self.dimensions,
                num_configs=self.n_configurations,
                max_configuration_size=self.max_configuration_size,
                mutables_mask=self.mutables,
                seed=self.seed
            )
            history = {tuple(c): [1.2] for c in configurations}
            #history = {}

            #scores = [math.inf for s in range(len(configurations))]
            #candidates = [None for c in range(len(configurations))]

            for i in range(self.hyperband_bracket + 1):
                budget = self.bracket_budget * pow(self.downsample, i)
                results = [self.process_one(candidate=None, idx=idx, configuration=configuration, budget=budget, history=history) for configuration in configurations]
                
                scores = [r[0] for r in results]
                candidates = [r[1] for r in results]
                top_indices = np.argsort(scores)[:max(int(len(scores) / self.downsample), 1)]
                configurations = [configurations[j] for j in top_indices]
                candidates = [candidates[j] for j in top_indices]
                scores = [scores[j] for j in top_indices]
                
                '''
                for score, candidate, configuration in tqdm(zip(scores, candidates, configurations), total=len(configurations), desc=f'Running Round {i} of SH. Evaluating {len(configurations)} configurations with budget of {budget}'):
                    new_score, new_candidate = self.objective.evaluate(
                        classifier=self.classifier,
                        configuration=configuration,
                        budget=budget,
                        x=self.x[idx],
                        y=self.y[idx],
                        eps=self.eps,
                        distance=self.distance,
                        features_min_max=self.features_min_max,
                        generate_perturbation=generate_perturbation,
                        history=history,
                        candidate=candidate
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
                #scores, candidates = zip(*results)
            scores, candidates = zip(*results)
            scores, candidates = list(scores), list(candidates)
            '''
            
                #print(f'scores of round {i} {scores}')
            #print(f'len scores {len(scores)}, len configs {len(configurations)}, len candidates {len(candidates)}')
            all_results.append((scores, configurations, candidates, history))
        return all_results
        
        


    
