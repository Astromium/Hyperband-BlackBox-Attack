import numpy as np
from typing import Tuple, Any

def generate_perturbation(shape: Tuple, eps: float, distance: Any):
    perturbation = np.random.rand(*shape)
    perturbation = (perturbation / np.linalg.norm(perturbation, ord=distance)) * eps
    return perturbation