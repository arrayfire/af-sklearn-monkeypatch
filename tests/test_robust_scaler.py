import sklearn
import sklearn.preprocessing
import time
import afsklearn
import numpy as np

nbench = 5
rng = np.random.RandomState(0)
X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(250, 4)), axis=0)

tic = time.perf_counter()
#import pdb; pdb.set_trace()
for n in range(nbench):
    qt = sklearn.preprocessing.RobustScaler()
    res = qt.fit_transform(X)
toc = time.perf_counter()
print(f"sklearn fit time {(toc - tic)/nbench:0.4f} seconds")

#
#afsklearn.patch_sklearn()
#tic = time.perf_counter()
#for n in range(nbench):
#    qt = sklearn.preprocessing.RobustScaler()
#    res_af = qt.fit_transform(X)
#toc = time.perf_counter()
#print(f"afsklearn fit time {(toc - tic)/nbench:0.4f} seconds")
#afsklearn.unpatch_sklearn()

#print(np.max(np.abs(res - res_af)))
