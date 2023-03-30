from succesive_halving import SuccessiveHalving
from typing import Any, List
from dataclasses import dataclass
from evaluators import Evaluator
from numpy.typing import NDArray
from sampler import Sampler
from worker import Worker
from joblib import Parallel, delayed
import os
import math



print(f'Hello {os.getpid()} from Hyperband')

def run_worker(args):
    sh = SuccessiveHalving(**args)
    scores, configurations, candidates = sh.run()
    return scores, configurations, candidates

@dataclass
class Hyperband():
    objective: Evaluator
    classifier_path: str
    x: NDArray
    y: int
    sampler: Sampler
    eps: float
    dimensions: int
    max_configuration_size: int
    R: int
    downsample: int
    distance: str


    def generate(self, mutables=None, features_min_max=None):
        if self.downsample <= 1:
            raise ValueError('Downsample must be > 1')
        
        all_scores, all_configurations, all_candidates = [], [], []
        # Number of Hyperband rounds
        s_max = math.floor(math.log(self.R, self.downsample))
        B = self.R * (s_max + 1)
        print(f'Hyperband brackets {s_max + 1}')
        params = [ 
            {'objective': self.objective, 
             'classifier_path': self.classifier_path, 
             'sampler': self.sampler, 
             'x': self.x, 'y': self.y, 
             'eps': self.eps, 
             'dimensions': self.dimensions, 
             'max_configuration_size': self.max_configuration_size, 
             'distance': self.distance, 
             'max_ressources_per_configuration': self.R,
             'downsample': self.downsample,
             'mutables': mutables,
             'features_min_max': features_min_max,
             'n_configurations': max(round((B * (self.downsample ** i)) / (self.R * (i + 1))), 1),
             'bracket_budget': max(round(self.R / (self.downsample ** i)), 1),
             'hyperband_bracket': i
            } 
            for i in reversed(range(s_max + 1)) 
        ]
        
        '''
        p = Pool(os.cpu_count())
        results = p.map(run_worker, params)
        p.close()
        p.join()
        '''
        
        results = Parallel(n_jobs=s_max+1, verbose=0, backend='multiprocessing', prefer='processes')(delayed(run_worker)(params[i]) for i in range(s_max + 1))
        

        for (scores, configurations, candidates) in results:
            all_scores.extend(scores)
            all_configurations.extend(configurations)
            all_candidates.extend(candidates)
        
        
        #res = processes[0].run()
        #res1 = processes[1].run()
        
        '''
        for p in processes:
            scores, configurations, candidates = p.run()
            with lock:
                all_scores.extend(scores)
                all_configurations.extend(configurations)
                all_candidates.extend(candidates)    
        '''

        '''
        for i in reversed(range(s_max + 1)):
            print(f'\n Starting Hyperband Bracket {i} \n')
            p = Worker(kwargs=params[i])
            n = round((B * (self.downsample ** i)) / (self.R * (i + 1)))
            n = max(n, 1)
            bracket_budget = max(round(self.R / (self.downsample ** i)), 1)
            sh = SuccessiveHalving(
                objective=self.objective,
                classifier=self.classifier, 
                sampler=self.sampler,
                x=self.x,
                y=self.y,
                eps=self.eps,
                dimensions=self.dimensions,
                max_configuration_size=self.max_configuration_size,
                distance=self.distance,
                max_ressources_per_configuration=self.R,
                downsample=self.downsample,
                bracket_budget=bracket_budget,
                n_configurations=n,
                mutables=mutables,
                features_min_max=features_min_max,
                hyperband_bracket=i
            )

            scores, configurations, candidates = p.run()

            all_scores.extend(scores)
            all_configurations.extend(configurations)
            all_candidates.extend(candidates)
            '''
        
        return all_scores, all_configurations, all_candidates


