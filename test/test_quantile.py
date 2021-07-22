import sklearn
import daal4py.sklearn
import time
import afsklearn
import numpy as np

nbench = 5
rng = np.random.RandomState(0)
X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25000, 100)), axis=0)

tic = time.perf_counter()
#import pdb; pdb.set_trace()
for n in range(nbench):
    qt = sklearn.preprocessing.QuantileTransformer()
    res = qt.fit_transform(X)
toc = time.perf_counter()
print(f"sklearn fit time {(toc - tic)/nbench:0.4f} seconds")

#daal4py.sklearn.patch_sklearn()
#tic = time.perf_counter()
#for n in range(nbench):
#    qt = sklearn.preprocessing.QuantileTransformer()
#    qt.fit_transform(X)
#toc = time.perf_counter()
#print(f"daal4py sklearn fit time {(toc - tic)/nbench:0.4f} seconds")
#
#daal4py.sklearn.unpatch_sklearn()


afsklearn.patch_sklearn()
tic = time.perf_counter()
for n in range(nbench):
    qt = sklearn.preprocessing.QuantileTransformer()
    res_af = qt.fit_transform(X)
toc = time.perf_counter()
print(f"afsklearn fit time {(toc - tic)/nbench:0.4f} seconds")
afsklearn.unpatch_sklearn()

print(np.max(np.abs(res - res_af)))
