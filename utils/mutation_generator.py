import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.mutation.pm import PolynomialMutation
from typing import Tuple
from numpy.typing import NDArray

def generate_mutations(pop_size: int, features_min_max: Tuple, x:NDArray):
    problem = Problem(n_var=len(features_min_max[0]), xl=np.array(features_min_max[0]), xu=np.array(features_min_max[1]))
    X = np.repeat(x[np.newaxis, :], repeats=pop_size, axis=0)
    pop = Population.new(X=X)
    mutation = PolynomialMutation(prob=1.0, eta=30)
    off = mutation.do(problem, pop)
    Xp = off.get('X')
    return Xp
