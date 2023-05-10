import numpy as np
from numpy.typing import NDArray
import random
from typing import Tuple, List

'''
def generate_perturbation(shape: Tuple, eps: float, distance: str):
    perturbation = np.random.rand(*shape)
    bound = random.random() * eps
    perturbation = (perturbation / np.linalg.norm(perturbation, ord=distance)) * bound
    return perturbation
'''


def generate_perturbation(configuration: List, features_min: List, features_max: List, x: NDArray):
    perturbation = [random.uniform(
        x[c] - features_max[c], features_max[c] - x[c]) for c in configuration]
    return perturbation

# pb = generate_perturbation(configuration=[0, 2, 4], features_max=(10, 20, 30, 40, 50), x=[1,2,3,4,5])
# print(f'pb {pb}')
