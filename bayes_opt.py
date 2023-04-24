# wrapper class for skopt.Optimizer
from skopt.optimizer import Optimizer
from utils.config_generator2 import ConfigGenerator
import timeit

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

cg = ConfigGenerator(mutable_features=list(range(63)))
bopt = BayesianOptimizer(cg=cg, n_jobs=-1)

def target(x):
    return sum(x)

history_x = []
history_y = []
configs = cg.get_configurations(n_sample=81, logits=None)
scores = [target(x[0]) for x in configs]
history_x = [x[1] for x in configs]
#bopt.update_optimizer(history_x, scores)
start = timeit.default_timer()
logits = bopt.get_next(n_samples=30)
end = timeit.default_timer()
print(f'took {(end - start) / 60}')


'''
for i in range(50):
    bopt = BayesianOptimizer(cg=cg, n_jobs=1)
    if i > 0:
        bopt.update_optimizer(history_x, history_y)
    start = timeit.default_timer()
    print(f'Iteration {i}')
    print('Sampling from a prior')
    logits = bopt.get_next(n_samples=30)
    print('Getting configs from logits')
    configs = cg.get_configurations(n_sample=30, logits=logits)
    end = timeit.default_timer()
    print(f'It took {(end - start) / 60}\n')
    scores = [target(x[0]) for x in configs]
    print('Updating history')
    for i in range(len(configs)):
        history_x.append(configs[i][1])
        history_y.append(scores[i])

'''