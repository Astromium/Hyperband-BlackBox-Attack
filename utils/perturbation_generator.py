import numpy as np
from typing import Tuple

def generate_perturbation(shape: Tuple, eps: float, distance: str):
    if distance == 'l2':
        perturbation = np.random.rand(*shape)
        perturbation = (perturbation / np.linalg.norm(perturbation, ord=2)) * eps
        return perturbation
    elif distance == 'inf':
        perturbation = np.random.rand(*shape)
        perturbation = (perturbation / np.linalg.norm(perturbation, ord=np.inf)) * eps
        return perturbation
    else:
        raise NotImplementedError()