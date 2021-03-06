# import time

# import numpy as np
# import sklearn

# import afsklearn

# nbench = 5
# # time fit for default sklearn.SVC
# print(sklearn.svm.SVC)
# X = np.random.rand(10000, 2)
# y = np.round(np.random.rand(10000)).astype(np.int32)
# clf = sklearn.svm.SVC(kernel='linear', C = 1.0)
# tic = time.perf_counter()
# for n in range(nbench):
#     clf.fit(X,y)
# toc = time.perf_counter()
# print(f"sklearn fit time {(toc - tic)/nbench:0.4f} seconds")


# # monkey patch afsklearn
# afsklearn.patch_sklearn()

# print(sklearn.svm.SVC)
# clf = sklearn.svm.SVC(kernel='linear', C = 1.0)
# tic = time.perf_counter()
# for n in range(nbench):
#     clf.fit(X,y)
# toc = time.perf_counter()
# print(f"arrayfire fit time {(toc - tic)/nbench:0.4f} seconds")

# afsklearn.unpatch_sklearn()
