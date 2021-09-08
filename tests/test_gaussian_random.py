import sklearn
import sklearn.preprocessing
import sklearn.random_projection
import time
import afsklearn
import numpy as np
import pickle
from afsklearn.patcher import Patcher


nbench = 1
#rng = np.random.RandomState(0)
#X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(548, 100000)), axis=0)

with open('training_data.pickle', 'rb') as f:
    X = pickle.load(f)


tic = time.perf_counter()
#import pdb; pdb.set_trace()
for n in range(nbench):
    transformer = sklearn.random_projection.GaussianRandomProjection()
    X_new = transformer.fit_transform(X)
    X_new.shape
toc = time.perf_counter()
print(f"sklearn fit time {(toc - tic)/nbench:0.4f} seconds")

Patcher.patch('gaussian_random_projection')
tic = time.perf_counter()
for n in range(nbench):
    transformer = sklearn.random_projection.GaussianRandomProjection(eps=0.1, random_state=0)
    X_new = transformer.fit_transform(X)
    X_new.shape
toc = time.perf_counter()
print(f"afsklearn fit time {(toc - tic)/nbench:0.4f} seconds")
Patcher.rollback('gaussian_random_projection')

#print(np.max(np.abs(res - res_af)))
