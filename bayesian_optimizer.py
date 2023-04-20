from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Callable
from numpy.typing import NDArray
import pandas as pd
import numpy as np
from utils.config_generator import ConfigGenerator
import sys
import warnings
warnings.filterwarnings(action='ignore')

class BayesianOptimizer():
    def __init__(self, target: Callable, x_init: NDArray, y_init: NDArray, n_iter: int, scale: int, batch_size: int, config_generator: ConfigGenerator) -> None:
        self.x_init = x_init
        self.y_init = y_init
        self.target = target
        self.n_iter = n_iter
        self.scale = scale
        self.batch_size = batch_size
        self.config_generator = config_generator
        self.surrogate = GaussianProcessRegressor()
        self.best_samples = pd.DataFrame(columns=['x', 'y', 'ei'])

    def _extend_prior(self, x: NDArray, y: NDArray):
        self.x_init = np.vstack((self.x_init, x[np.newaxis, :]))
        self.y_init = np.hstack((self.y_init, y))

    def _get_ei(self, x_new):
        mu_new, std_new = self.surrogate.predict(x_new[np.newaxis, :], return_std=True)
        std_new = std_new.reshape(-1, 1)
        if std_new == 0.0:
            return 0.0
        
        mu = self.surrogate.predict(self.x_init)
        mu_max = np.max(mu)
        z = (mu_new - mu_max) / std_new
        ei = (mu_new - mu_max) * norm.cdf(z) + std_new * norm.pdf(z)
        return ei
    
    def _acquisition_fn(self, x):
        return self._get_ei(x)

    def _get_next_point(self):
        min_ei = float(sys.maxsize)
        x_optimal = None

        for _, logit in self.config_generator.get_configurations(n_sample=self.batch_size, logits=None):
            bounds = [(0, 1) for _ in range(len(self.config_generator.mutable_features))]
            bounds.append((1, len(self.config_generator.mutable_features)))
            
            response = minimize(fun=self._acquisition_fn, x0=logit, bounds=tuple(bounds), method='L-BFGS-B')
            if response.fun < min_ei:
                min_ei = response.fun
                x_optimal = response.x
            
            return x_optimal, min_ei
        
    def optimize(self):
        y_max_idx = np.argmax(self.y_init)
        best_y = self.y_init[y_max_idx]
        best_x = self.x_init[y_max_idx]
        best_ei = None
        for i in range(self.n_iter):
            self.surrogate.fit(self.x_init, self.y_init)
            x_next, ei = self._get_next_point()
            y_next = self.target(x_next)
            self._extend_prior(x_next, y_next)

            if y_next < best_y:
                best_y = y_next
                best_x = np.copy(x_next)
                best_ei = ei
            
            self.best_samples = self.best_samples.append({'y': best_y, 'ei': best_ei}, ignore_index=True)
        return best_x, best_y
    
cg = ConfigGenerator(mutable_features=list(range(8)))

def target(logit):
    config, _ = cg.get_configurations(n_sample=1, logits=logit)
    #print(config, np.sum(config))
    return np.sum(config)

x = np.array(cg.get_samples(n_sample=81))
y = np.array([target(s) for s in x])

bopt = BayesianOptimizer(target=target, x_init=x, y_init=y, n_iter=10, scale=1, batch_size=100, config_generator=cg)
for i in range(81):
    x_new, y_new = bopt.optimize()
    print(f'Y_new {y_new}')
print(f'X {x_new}, Y {y_new}')
import matplotlib.pyplot as plt
plt.plot(list(range(len(bopt.best_samples['y']))), bopt.best_samples['y'])
plt.plot(list(range(len(bopt.best_samples['ei']))), bopt.best_samples['ei'])
plt.show()
print(bopt.best_samples['ei'])
