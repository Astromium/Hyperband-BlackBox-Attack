import numpy as np
from skopt import Optimizer
from skopt.acquisition import gaussian_ei
from utils.config_generator2 import ConfigGenerator

def objective(x):
    return np.sum(np.sin(x))

cg = ConfigGenerator(list(range(8)))
print(f'space {cg.space}')
opt = Optimizer(
    dimensions=cg.space,
    base_estimator='gp',
    acq_func='EI',
    acq_optimizer='sampling',
    n_jobs=-1,
    random_state=42
)

for i in range(20):
    next_hp = opt.ask()
    obj = objective(next_hp)
    res = opt.tell(next_hp, obj)
    

best_hp = opt.Xi
best_obj = opt.yi

#print(f'best_hp {best_hp}')
print(f'best_obj {best_obj}')

hps = opt.ask(n_points=10)
res = [objective(h) for h in hps]
print(f'res {res}')
opt.tell(hps, res)

hps = opt.ask(n_points=4)
res = [objective(h) for h in hps]
print(f'res2 {res}')
opt.tell(hps, res)

hps = opt.ask(n_points=4)
res = [objective(h) for h in hps]
print(f'res3 {res}')

opt.tell(hps, res)

hps = opt.ask(n_points=4)
res = [objective(h) for h in hps]
print(f'res4{res}')


