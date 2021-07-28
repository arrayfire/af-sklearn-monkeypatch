import numpy as np
import scipy.sparse as sp
from sklearn.utils.sparsefuncs import _raise_error_wrong_axis
from sklearn.utils.sparsefuncs_fast import csc_mean_variance_axis0 as _csc_mean_var_axis0
from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0 as _csr_mean_var_axis0


def mean_variance_axis(X, axis, weights=None, return_sum_weights=False):
    """Compute mean and variance along an axis on a CSR or CSC matrix.
    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It can be of CSR or CSC format.
    axis : {0, 1}
        Axis along which the axis should be computed.
    weights : ndarray of shape (n_samples,) or (n_features,), default=None
        if axis is set to 0 shape is (n_samples,) or
        if axis is set to 1 shape is (n_features,).
        If it is set to None, then samples are equally weighted.
        .. versionadded:: 0.24
    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature
        if `axis=0` or each sample if `axis=1`.
        .. versionadded:: 0.24
    Returns
    -------
    means : ndarray of shape (n_features,), dtype=floating
        Feature-wise means.
    variances : ndarray of shape (n_features,), dtype=floating
        Feature-wise variances.
    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if `return_sum_weights` is `True`.
    """
    _raise_error_wrong_axis(axis)

    if isinstance(X, sp.csr_matrix):
        if axis == 0:
            return _csr_mean_var_axis0(
                X, weights=weights, return_sum_weights=return_sum_weights)
        else:
            return _csc_mean_var_axis0(
                X.T, weights=weights, return_sum_weights=return_sum_weights)
    elif isinstance(X, sp.csc_matrix):
        if axis == 0:
            return _csc_mean_var_axis0(
                X, weights=weights, return_sum_weights=return_sum_weights)
        else:
            return _csr_mean_var_axis0(
                X.T, weights=weights, return_sum_weights=return_sum_weights)
    else:
        _raise_typeerror(X)


def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.
    This function is used to support sparse matrices; it modifies data
    in-place.
    """
    n_elems = len(data) + n_zeros
    if not n_elems:
        return np.nan
    n_negative = np.count_nonzero(data < 0)
    middle, is_odd = divmod(n_elems, 2)
    data.sort()

    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)

    return (_get_elem_at_rank(middle - 1, data, n_negative, n_zeros) +
            _get_elem_at_rank(middle, data, n_negative, n_zeros)) / 2.


def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    """Find the value in data augmented with n_zeros for the given rank"""
    if rank < n_negative:
        return data[rank]
    if rank - n_negative < n_zeros:
        return 0
    return data[rank - n_zeros]


def min_max_axis(X, axis, ignore_nan=False):
    """Compute minimum and maximum along an axis on a CSR or CSC matrix and
    optionally ignore NaN values.
    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It should be of CSR or CSC format.
    axis : {0, 1}
        Axis along which the axis should be computed.
    ignore_nan : bool, default=False
        Ignore or passing through NaN values.
        .. versionadded:: 0.20
    Returns
    -------
    mins : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Feature-wise minima.
    maxs : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Feature-wise maxima.
    """
    if isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix):
        if ignore_nan:
            return _sparse_nan_min_max(X, axis=axis)
        else:
            return _sparse_min_max(X, axis=axis)
    else:
        _raise_typeerror(X)


def _min_or_max_axis(X, axis, min_or_max):
    N = X.shape[axis]
    if N == 0:
        raise ValueError("zero-size array to reduction operation")
    M = X.shape[1 - axis]
    mat = X.tocsc() if axis == 0 else X.tocsr()
    mat.sum_duplicates()
    major_index, value = _minor_reduce(mat, min_or_max)
    not_full = np.diff(mat.indptr)[major_index] < N
    value[not_full] = min_or_max(value[not_full], 0)
    mask = value != 0
    major_index = np.compress(mask, major_index)
    value = np.compress(mask, value)

    if axis == 0:
        res = sp.coo_matrix((value, (np.zeros(len(value)), major_index)),
                            dtype=X.dtype, shape=(1, M))
    else:
        res = sp.coo_matrix((value, (major_index, np.zeros(len(value)))),
                            dtype=X.dtype, shape=(M, 1))
    return res.A.ravel()


def _sparse_min_or_max(X, axis, min_or_max):
    if axis is None:
        if 0 in X.shape:
            raise ValueError("zero-size array to reduction operation")
        zero = X.dtype.type(0)
        if X.nnz == 0:
            return zero
        m = min_or_max.reduce(X.data.ravel())
        if X.nnz != np.product(X.shape):
            m = min_or_max(zero, m)
        return m
    if axis < 0:
        axis += 2
    if (axis == 0) or (axis == 1):
        return _min_or_max_axis(X, axis, min_or_max)
    else:
        raise ValueError("invalid axis, use 0 for rows, or 1 for columns")


def _sparse_min_max(X, axis):
    return (_sparse_min_or_max(X, axis, np.minimum),
            _sparse_min_or_max(X, axis, np.maximum))


def _sparse_nan_min_max(X, axis):
    return(_sparse_min_or_max(X, axis, np.fmin),
           _sparse_min_or_max(X, axis, np.fmax))


def _raise_typeerror(X):
    """Raises a TypeError if X is not a CSR or CSC matrix"""
    input_type = X.format if sp.issparse(X) else type(X)
    err = "Expected a CSR or CSC sparse matrix, got %s." % input_type
    raise TypeError(err)


def _minor_reduce(X, ufunc):
    major_index = np.flatnonzero(np.diff(X.indptr))

    # reduceat tries casts X.indptr to intp, which errors
    # if it is int64 on a 32 bit system.
    # Reinitializing prevents this where possible, see #13737
    X = type(X)((X.data, X.indices, X.indptr), shape=X.shape)
    value = ufunc.reduceat(X.data, X.indptr[major_index])
    return major_index, value
