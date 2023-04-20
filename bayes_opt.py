# wrapper class for skopt.Optimizer
from skopt.optimizer import Optimizer
from utils.config_generator2 import ConfigGenerator

class BayesianOptimizer():
    def __init__(self, cg: ConfigGenerator, base_estimator: str = 'gp', acq_func: str = 'EI', acq_optimizer: str = 'sampling', n_jobs: int = 1) -> None:
        self.cg = cg
        self.base_estimator = base_estimator
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.n_jobs = n_jobs
        self.optimizer = Optimizer(
            dimensions=self.cg.space,
            base_estimator=self.base_estimator,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            n_jobs=self.n_jobs
        )
    
    def get_next(self, n_samples: int = 1):
        return self.optimizer.ask(n_points=n_samples)
    
    def update_optimizer(self, x, y):
        self.optimizer.tell(x, y)

