import os
import numpy as np
import pickle
import time
from functools import wraps
import arrayfire as af

def measure_time(func):
    @wraps(func)
    def time_it(*args, **kwargs):
        print(f'Testing "{func.__name__}"')
        start_timer = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end_timer = time.perf_counter()
            print(f"Execution time: {end_timer - start_timer} sec")
    return time_it

from afsklearn.patcher import Patcher

import sklearn
from sklearn.neural_network import MLPClassifier

#Patcher.patch("mlp_classifier")

with open('mlp_training_inputs.pickle', 'rb') as f:
   X = pickle.load(f)

with open('mlp_training_output.pickle', 'rb') as f:
   y = pickle.load(f)

#with open('mlp_hyperparams.pickle', 'rb') as f:
   #hyperparams = pickle.load(f)


#clf = sklearn.neural_network.MLPClassifier(
#      hidden_layer_sizes=(100,),
#      activation='relu',
#      solver='adam',
#      learning_rate='constant',
#      learning_rate_init=0.001,
#      power_t=0.5,
#      shuffle=True,
#      momentum=0.9,
#      nesterovs_momentum=True,
#      early_stopping=False,
#      beta_1=0.9,
#      beta_2=0.999,
#      epsilon=1e-08,
#      n_iter_no_change=10,
#      max_fun=15000,
#      alpha=0.0001,
#      batch_size='auto',
#      max_iter=200,
#      tol=0.0001,
#      verbose=True,
#      random_state=0
#)
#clf = MLPClassifier(
#      hidden_layer_sizes=hyperparams['hidden_layer_sizes'],
#      activation=hyperparams['activation'],
#      solver=hyperparams['solver']['choice'],
#      learning_rate=hyperparams['solver'].get('learning_rate', 'constant'),
#      learning_rate_init=hyperparams['solver'].get('learning_rate_init', 0.001),
#      power_t=hyperparams['solver'].get('power_t', 0.5),
#      shuffle=hyperparams['solver'].get('shuffle', True),
#      momentum=hyperparams['solver'].get('momentum', 0.9),
#      nesterovs_momentum=hyperparams['solver'].get('nesterovs_momentum', True),
#      early_stopping=hyperparams['solver'].get('early_stopping', False),
#      beta_1=hyperparams['solver'].get('beta_1', 0.9),
#      beta_2=hyperparams['solver'].get('beta_2', 0.999),
#      epsilon=hyperparams['solver'].get('epsilon', 1e-08),
#      n_iter_no_change=hyperparams['solver'].get('n_iter_no_change', 10),
#      max_fun=hyperparams['solver'].get('max_fun', 15000),
#      alpha=hyperparams['alpha'],
#      batch_size=hyperparams['batch_size'],
#      max_iter=hyperparams['max_iter'],
#      tol=hyperparams['tol'],
#      warm_start=hyperparams['warm_start'],
#      validation_fraction=hyperparams['validation_fraction'],
#      verbose=True,
#      random_state=0
#)

#import pdb; pdb.set_trace();
#clf.fit(X, y)

#Patcher.rollback("mlp_classifier")

def sklearn_example() -> None:
    from sklearn.neural_network import MLPClassifier
    clf = sklearn.neural_network.MLPClassifier(
          hidden_layer_sizes=(100,),
          activation='relu',
          #activation='identity',
          #activation='logistic',
          #activation='tanh',
          solver='adam',
          #solver='sgd',
          #solver='lbfgs',
          learning_rate='constant',
          learning_rate_init=0.001,
          power_t=0.5,
          shuffle=True,
          momentum=0.9,
          nesterovs_momentum=True,
          early_stopping=False,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-08,
          n_iter_no_change=10,
          max_fun=15000,
          alpha=0.0001,
          batch_size='auto',
          max_iter=100,
          tol=0.0001,
          verbose=True,
          random_state=0
    )
    clf.fit(X, y)
    clf.predict(X)
    print(f"Score: {clf.score(X, y)}")


@measure_time
def test_sklearn_mlp() -> None:
    sklearn_example()


@measure_time
def test_afsklearn_mlp() -> None:
    Patcher.patch("mlp_classifier")
    sklearn_example()
    Patcher.rollback("mlp_classifier")


if __name__ == "__main__":
    af.info()
    test_afsklearn_mlp()
    test_sklearn_mlp()
