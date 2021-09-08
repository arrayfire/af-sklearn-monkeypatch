import sklearn
import sklearn.preprocessing
import sklearn.random_projection
import time
import afsklearn
import numpy as np
import pickle
from afsklearn.patcher import Patcher

def bench_gaussian_random_projection(nbench=1):

    with open('training_data.pickle', 'rb') as f:
        X = pickle.load(f)

    tic = time.perf_counter()
    for n in range(nbench):
        transformer = sklearn.random_projection.GaussianRandomProjection()
        X_new = transformer.fit_transform(X)
        print(X_new.shape)
    toc = time.perf_counter()
    sklearn_time = toc - tic
    print(f"sklearn fit time {(sklearn_time)/nbench:0.4f} seconds")


    Patcher.patch('gaussian_random_projection')
    tic = time.perf_counter()
    for n in range(nbench):
        transformer = sklearn.random_projection.GaussianRandomProjection(eps=0.1, random_state=0)
        X_new_af = transformer.fit_transform(X)
        print(X_new_af.shape)
    toc = time.perf_counter()
    afsklearn_time = toc - tic
    print(f"afsklearn fit time {(afsklearn_time)/nbench:0.4f} seconds")
    Patcher.rollback('gaussian_random_projection')

    print(np.max(np.abs(X_new - X_new_af)))
    return sklearn_time/nbench, afsklearn_time/nbench

def 
