import sklearn
import sklearn.preprocessing
import sklearn.random_projection

from sklearn.metrics import euclidean_distances
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal

import time
import afsklearn

import numpy as np
import numpy.random
from numpy.random import RandomState
import scipy.sparse as sp
import pickle
from pathlib import Path
from afsklearn.patcher import Patcher
import pytest


from sklearn.random_projection import GaussianRandomProjection
from afsklearn.random_projection import GaussianRandomProjection as afGaussianRandomProjection

all_RandomProjection = [GaussianRandomProjection, afGaussianRandomProjection]
nbench = 1

def make_sparse_random_data(n_samples, n_features, n_nonzeros, random_state=None):
    rng = np.random.RandomState(random_state)
    data_coo = sp.coo_matrix(
        (
            rng.randn(n_nonzeros),
            (
                rng.randint(n_samples, size=n_nonzeros),
                rng.randint(n_features, size=n_nonzeros),
            ),
        ),
        shape=(n_samples, n_features),
    )
    return data_coo.toarray(), data_coo.tocsr()

def densify(matrix):
    if not sp.issparse(matrix):
        return matrix
    else:
        return matrix.toarray()


n_samples, n_features = (10, 1000)
n_nonzeros = int(n_samples * n_features / 100.0)
data, data_csr = make_sparse_random_data(n_samples, n_features, n_nonzeros)

@pytest.mark.parametrize("n_components, fit_data", [("auto", [[0, 1, 2]]), (-10, data)])
def test_random_projection_transformer_invalid_input(n_components, fit_data):
    for RandomProjection in all_RandomProjection:
        with pytest.raises(ValueError):
            RandomProjection(n_components=n_components).fit(fit_data)


def test_try_to_transform_before_fit():
    for RandomProjection in all_RandomProjection:
        with pytest.raises(ValueError):
            RandomProjection(n_components="auto").transform(data)


def test_too_many_samples_to_find_a_safe_embedding():
    data, _ = make_sparse_random_data(1000, 100, 1000)

    for RandomProjection in all_RandomProjection:
        rp = RandomProjection(n_components="auto", eps=0.1)
        expected_msg = (
            "eps=0.100000 and n_samples=1000 lead to a target dimension"
            " of 5920 which is larger than the original space with"
            " n_features=100"
        )
        with pytest.raises(ValueError, match=expected_msg):
            rp.fit(data)

def test_random_projection_embedding_quality():
    data, _ = make_sparse_random_data(8, 5000, 15000)
    eps = 0.2

    original_distances = euclidean_distances(data, squared=True)
    original_distances = original_distances.ravel()
    non_identical = original_distances != 0.0

    # remove 0 distances to avoid division by 0
    original_distances = original_distances[non_identical]

    for RandomProjection in all_RandomProjection:
        rp = RandomProjection(n_components="auto", eps=eps, random_state=0)
        projected = rp.fit_transform(data)

        projected_distances = euclidean_distances(projected, squared=True)
        projected_distances = projected_distances.ravel()

        # remove 0 distances to avoid division by 0
        projected_distances = projected_distances[non_identical]

        distances_ratio = projected_distances / original_distances

        # check that the automatically tuned values for the density respect the
        # contract for eps: pairwise distances are preserved according to the
        # Johnson-Lindenstrauss lemma
        print(f'{distances_ratio.max()} < {1 + eps} ')
        print(f'{1 - eps} < {distances_ratio.min()}')
        assert distances_ratio.max() < 1 + eps
        assert 1 - eps < distances_ratio.min()

def test_correct_RandomProjection_dimensions_embedding():
    for RandomProjection in all_RandomProjection:
        rp = RandomProjection(n_components="auto", random_state=0, eps=0.5).fit(data)

        # the number of components is adjusted from the shape of the training
        # set
        assert rp.n_components == "auto"
        assert rp.n_components_ == 110

        assert rp.components_.shape == (110, n_features)

        projected_1 = rp.transform(data)
        assert projected_1.shape == (n_samples, 110)

        # once the RP is 'fitted' the projection is always the same
        projected_2 = rp.transform(data)
        assert_array_equal(projected_1, projected_2)

        # fit transform with same random seed will lead to the same results
        rp2 = RandomProjection(random_state=0, eps=0.5)
        projected_3 = rp2.fit_transform(data)
        assert_array_equal(projected_1, projected_3)

        # Try to transform with an input X of size different from fitted.
        with pytest.raises(ValueError):
            rp.transform(data[:, 1:5])


def test_warning_n_components_greater_than_n_features():
    n_features = 20
    data, _ = make_sparse_random_data(5, n_features, int(n_features / 4))

    for RandomProjection in all_RandomProjection:
        with pytest.warns(UserWarning):
            RandomProjection(n_components=n_features + 1).fit(data)


def test_works_with_sparse_data():
    n_features = 20
    data, _ = make_sparse_random_data(5, n_features, int(n_features / 4))

    for RandomProjection in all_RandomProjection:
        rp_dense = RandomProjection(n_components=3, random_state=1).fit(data)
        rp_sparse = RandomProjection(n_components=3, random_state=1).fit(
            sp.csr_matrix(data)
        )
        assert_array_almost_equal(
            densify(rp_dense.components_), densify(rp_sparse.components_)
        )


#assume test data in same directory as test file
try:
    dirpath = Path().absolute().as_posix() + '/'
    print(dirpath + 'training_data.pickle')

    with open(dirpath + 'training_data.pickle', 'rb') as f:
        X = pickle.load(f)
        print(X.shape)
except:
    rng = np.random.RandomState(0)
    X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(548, 100000)), axis=0)
    print(X.shape)

def test_af_speedup():
    tic = time.perf_counter()
    for n in range(nbench):
        transformer = sklearn.random_projection.GaussianRandomProjection()
        X_new = transformer.fit_transform(X)
        X_new.shape
    toc = time.perf_counter()
    print(f"sklearn fit time {(toc - tic)/nbench:0.4f} seconds")
    t_sklearn = (toc - tic)/nbench

    Patcher.patch('gaussian_random_projection')
    tic = time.perf_counter()
    for n in range(nbench):
        transformer = sklearn.random_projection.GaussianRandomProjection(eps=0.1, random_state=0)
        X_new = transformer.fit_transform(X)
        X_new.shape
    toc = time.perf_counter()
    print(f"afsklearn fit time {(toc - tic)/nbench:0.4f} seconds")
    t_afsklearn = (toc - tic)/nbench
    Patcher.rollback('gaussian_random_projection')
    #print(np.max(np.abs(res - res_af)))

    print(f'{t_afsklearn}  <? {t_sklearn}')
    assert t_afsklearn < t_sklearn

