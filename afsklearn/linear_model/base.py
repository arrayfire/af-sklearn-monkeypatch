import numbers
from abc import ABCMeta, abstractmethod

import numpy as np  # FIXME
import scipy.sparse as sp
from scipy import sparse
from sklearn.linear_model._base import SPARSE_INTERCEPT_DECAY
from sklearn.utils._seq_dataset import ArrayDataset32, ArrayDataset64, CSRDataset32, CSRDataset64
from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils.validation import FLOAT_DTYPES

from .._extmath import safe_sparse_dot
from .._sparsefuncs import mean_variance_axis
from .._validation import check_array, check_is_fitted, check_random_state
from ..base import afBaseEstimator
from ..preprocessing._data import normalize as f_normalize


def _preprocess_data(X, y, fit_intercept, normalize=False, copy=True,
                     sample_weight=None, return_mean=False, check_input=True):
    """Center and scale data.
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output
        X = (X - X_offset) / X_scale
    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).
    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=['csr', 'csc'], dtype=FLOAT_DTYPES)
    elif copy:
        if sp.issparse(X):
            X = X.copy()
        else:
            X = X.copy(order='K')

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0)
            if not return_mean:
                X_offset[:] = X.dtype.type(0)

            if normalize:

                # TODO: f_normalize could be used here as well but the function
                # inplace_csr_row_normalize_l2 must be changed such that it
                # can return also the norms computed internally

                # transform variance to norm in-place
                X_var *= X.shape[0]
                X_scale = np.sqrt(X_var, X_var)
                del X_var
                X_scale[X_scale == 0] = 1
                inplace_column_scale(X, 1. / X_scale)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)

        else:
            X_offset = np.average(X, axis=0, weights=sample_weight)
            X -= X_offset
            if normalize:
                X, X_scale = f_normalize(X, axis=0, copy=False, return_norm=True)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


class afLinearModel(afBaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _decision_function(self, X):
        check_is_fitted(self)

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    def predict(self, X):
        """
        Predict using the linear model.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    _preprocess_data = staticmethod(_preprocess_data)

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.

    def _more_tags(self):
        return {'requires_y': True}


def make_dataset(X, y, sample_weight, random_state=None):
    """Create ``Dataset`` abstraction for sparse and dense inputs.
    This also returns the ``intercept_decay`` which is different
    for sparse datasets.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples, )
        Target values.
    sample_weight : numpy array of shape (n_samples,)
        The weight of each sample
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    dataset
        The ``Dataset`` abstraction
    intercept_decay
        The intercept decay
    """

    rng = check_random_state(random_state)
    # seed should never be 0 in SequentialDataset64
    seed = rng.randint(1, np.iinfo(np.int32).max)

    if X.dtype == np.float32:
        CSRData = CSRDataset32
        ArrayData = ArrayDataset32
    else:
        CSRData = CSRDataset64
        ArrayData = ArrayDataset64

    if sp.issparse(X):
        dataset = CSRData(X.data, X.indptr, X.indices, y, sample_weight, seed=seed)
        intercept_decay = SPARSE_INTERCEPT_DECAY
    else:
        X = np.ascontiguousarray(X)
        dataset = ArrayData(X, y, sample_weight, seed=seed)
        intercept_decay = 1.0

    return dataset, intercept_decay


def _rescale_data(X, y, sample_weight):
    """Rescale data sample-wise by square root of sample_weight.
    For many linear models, this enables easy support for sample_weight.
    Returns
    -------
    X_rescaled : {array-like, sparse matrix}
    y_rescaled : {array-like, sparse matrix}
    """
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y
