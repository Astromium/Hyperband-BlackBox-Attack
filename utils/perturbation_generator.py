import numpy as np
import random
from typing import Tuple

def generate_perturbation(shape: Tuple, eps: float, distance: str):
    perturbation = np.random.rand(*shape)
    bound = random.random() * eps
    perturbation = (perturbation / np.linalg.norm(perturbation, ord=distance)) * bound
    return perturbation