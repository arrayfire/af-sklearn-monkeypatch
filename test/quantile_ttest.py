import sklearn
import daal4py.sklearn
import time
import afsklearn
import numpy as np
from afsklearn.preprocessing import QuantileTransformer as afQT

nbench = 5
rng = np.random.RandomState(0)
X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25000, 100)), axis=0)

tic = time.perf_counter()
#import pdb; pdb.set_trace()
for n in range(nbench):
    qt = sklearn.preprocessing.QuantileTransformer()
    qt.fit_transform(X)
toc = time.perf_counter()
print(f"sklearn fit time {(toc - tic)/nbench:0.4f} seconds")

sklearn.preprocessing.QuantileTransformer = afQT

for n in range(nbench):
    qt = sklearn.preprocessing.QuantileTransformer()
    qt.fit_transform(X)
toc = time.perf_counter()
print(f"sklearn fit time {(toc - tic)/nbench:0.4f} seconds")
