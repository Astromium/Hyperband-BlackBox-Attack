import numpy as np
from typing import Tuple

def generate_perturbation(shape: Tuple, eps: float, distance: str):
    perturbation = np.random.rand(*shape)
    perturbation = (perturbation / np.linalg.norm(perturbation, ord=distance)) * eps
    return perturbation