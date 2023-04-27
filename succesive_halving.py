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
    seed: int
    hyperband_bracket: int

    def run(self):
        if (self.downsample <= 1):
            raise (ValueError('Downsample must be > 1'))

        def round_n(n): return max(int(n), 1)

        all_results = []

        for idx in range(self.x.shape[0]):

            configurations = self.sampler.sample(
                dimensions=self.dimensions,
                num_configs=self.n_configurations,
                max_configuration_size=self.max_configuration_size,
                mutables_mask=self.mutables,
                seed=self.seed
            )

            scores = [math.inf for s in range(len(configurations))]
            candidates = [None for c in range(len(configurations))]

            results = []

            for i in range(self.hyperband_bracket + 1):
                budget = self.bracket_budget * pow(self.downsample, i)
                scores2 = []
                candidates2 = []
                for configuration in tqdm(configurations, total=len(configurations), desc=f'Running Round {i} of SH. Evaluating {len(configurations)} configurations with budget of {budget}'):
                    score, candidate = self.objective.evaluate(
                        classifier=self.classifier,
                        configuration=configuration,
                        budget=budget,
                        x=self.x[idx],
                        y=self.y[idx],
                        eps=self.eps,
                        distance=self.distance,
                        features_min_max=self.features_min_max,
                        generate_perturbation=generate_perturbation
                    )
                    scores2.append(score)
                    candidates2.append(candidate)
                top_indices = np.argsort(
                    scores2)[:int(len(scores2) / self.downsample)]
                # update configurations list
                # print(f'top_indices {top_indices}')
                # print(f'len configurations {len(configurations)}')
                configurations = [configurations[j] for j in top_indices]

            #     for score, candidate, configuration in tqdm(zip(scores, candidates, configurations), total=len(configurations), desc=f'Running Round {i} of SH. Evaluating {len(configurations)} configurations with budget of {budget}'):
            #         new_score, new_candidate = self.objective.evaluate(
            #             classifier=self.classifier,
            #             configuration=configuration,
            #             budget=budget,
            #             x=self.x[idx],
            #             y=self.y[idx],
            #             eps=self.eps,
            #             distance=self.distance,
            #             features_min_max=self.features_min_max,
            #             generate_perturbation=generate_perturbation
            #         )
            #         results.append(tuple([new_score, new_candidate]))
            #         # if new_score < score:
            #         #     results.append(tuple([new_score, new_candidate]))
            #         # else:
            #         #     results.append(tuple([score, candidate]))

            #     # Sort by minimum score
            #     results = sorted(zip(results, configurations),
            #                      key=lambda k: k[0][0])
            #     # keep the best half
            #     results = results[:round_n(
            #         len(configurations) / self.downsample)]
            #     results, configurations = zip(*results)
            #     # both arrays get casted to tuples for some reason
            #     results, configurations = list(results), list(configurations)
            # scores, candidates = zip(*results)
            # scores, candidates = list(scores), list(candidates)

            all_results.append((scores2, configurations, candidates2))

        return all_results
