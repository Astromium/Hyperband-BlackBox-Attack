import time
import numpy as np
import warnings


seed = 123
np.random.seed(seed)
warnings.filterwarnings('ignore')

min_budget, max_budget = 2, 50

import ConfigSpace as CS


def create_search_space(seed=123):
    """Parameter space to be optimized --- contains the hyperparameters
    """
    cs = CS.ConfigurationSpace(seed=seed)

    cs.add_hyperparameters([
        CS.UniformIntegerHyperparameter(
            'max_depth', lower=1, upper=15, default_value=2, log=False
        ),
        CS.UniformIntegerHyperparameter(
            'min_samples_split', lower=2, upper=128, default_value=2, log=True
        ),
        CS.UniformFloatHyperparameter(
            'max_features', lower=0.1, upper=0.9, default_value=0.5, log=False
        ),
        CS.UniformIntegerHyperparameter(
            'min_samples_leaf', lower=1, upper=64, default_value=1, log=True
        ),
    ])
    return cs

cs = create_search_space(seed)
print(cs)

dimensions = len(cs.get_hyperparameters())
print("Dimensionality of search space: {}".format(dimensions))

from sklearn.datasets import load_iris, load_digits, load_wine


classification = {"iris": load_iris, "digits": load_digits, "wine": load_wine}

from sklearn.model_selection import train_test_split


def prepare_dataset(model_type="classification", dataset=None):

    if model_type == "classification":
        if dataset is None:
            dataset = np.random.choice(list(classification.keys())) 
        _data = classification[dataset]()
    else:
        if dataset is None:
            dataset = np.random.choice(list(regression.keys()))
        _data = regression[dataset]()

    train_X, test_X, train_y, test_y = train_test_split(
        _data.get("data"), 
        _data.get("target"), 
        test_size=0.1, 
        shuffle=True, 
        random_state=seed
    )
    train_X, valid_X, train_y, valid_y = train_test_split(
        _data.get("data"), 
        _data.get("target"), 
        test_size=0.3, 
        shuffle=True, 
        random_state=seed
    )
    return train_X, train_y, valid_X, valid_y, test_X, test_y, dataset

train_X, train_y, valid_X, valid_y, test_X, test_y, dataset = \
    prepare_dataset(model_type="classification")

print(dataset)
print("Train size: {}\nValid size: {}\nTest size: {}".format(
    train_X.shape, valid_X.shape, test_X.shape
))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer


accuracy_scorer = make_scorer(accuracy_score)

def target_function(config, budget, **kwargs):
    # Extracting support information
    seed = kwargs["seed"]
    train_X = kwargs["train_X"]
    train_y = kwargs["train_y"]
    valid_X = kwargs["valid_X"]
    valid_y = kwargs["valid_y"]
    max_budget = kwargs["max_budget"]
    
    if budget is None:
        budget = max_budget
    
    start = time.time()
    # Building model 
    model = RandomForestClassifier(
        **config.get_dictionary(),
        n_estimators=int(budget),
        bootstrap=True,
        random_state=seed,
    )
    # Training the model on the complete training set
    model.fit(train_X, train_y)
    
    # Evaluating the model on the validation set
    valid_accuracy = accuracy_scorer(model, valid_X, valid_y)
    cost = time.time() - start
    
    # Evaluating the model on the test set as additional info
    test_accuracy = accuracy_scorer(model, test_X, test_y)
    
    result = {
        "fitness": -valid_accuracy,  # DE/DEHB minimizes
        "cost": cost,
        "info": {
            "test_score": test_accuracy,
            "budget": budget
        }
    }
    return result

from dehb import DEHB

dehb = DEHB(
    f=target_function, 
    cs=cs, 
    dimensions=dimensions, 
    min_budget=min_budget, 
    max_budget=max_budget,
    n_workers=1,
    output_path="./temp"
)

trajectory, runtime, history = dehb.run(
    total_cost=10,
    verbose=False,
    save_intermediate=False,
    # parameters expected as **kwargs in target_function is passed here
    seed=123,
    train_X=train_X,
    train_y=train_y,
    valid_X=valid_X,
    valid_y=valid_y,
    max_budget=dehb.max_budget
)

print(len(trajectory), len(runtime), len(history), end="\n\n")

# Last recorded function evaluation
last_eval = history[-1]
config, score, cost, budget, _info = last_eval

print("Last evaluated configuration, ")
print(dehb.vector_to_configspace(config), end="")
print("got a score of {}, was evaluated at a budget of {:.2f} and "
      "took {:.3f} seconds to run.".format(score, budget, cost))
print("The additional info attached: {}".format(_info))