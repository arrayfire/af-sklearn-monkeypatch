import arrayfire as af
import numpy as np
import scipy.sparse as sparse
from sklearn.utils.sparsefuncs_fast import csr_row_norms
from sklearn.utils.validation import _deprecate_positional_args


def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.
    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.
    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if isinstance(a, af.Array) or isinstance(b, af.Array):
        #ret = af.blas.matmul(a.as_type(af.Dtype.f32), b.as_type(af.Dtype.f32))

        if not isinstance(a, af.Array):
            a = af.interop.from_ndarray(a)
        if not isinstance(b, af.Array):
            b = af.interop.from_ndarray(b)

        #TODO: check a&b type()?
        if a.type() == af.Dtype.f64 or b.type() == af.Dtype.f64:
            ret = af.blas.matmul(a.as_type(af.Dtype.f64), b.as_type(af.Dtype.f64))
        else:
            ret = af.blas.matmul(a.as_type(af.Dtype.f32), b.as_type(af.Dtype.f32))
    else:
        if a.ndim > 2 or b.ndim > 2:
            if sparse.issparse(a):
                # sparse is always 2D. Implies b is 3D+
                # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
                b_ = np.rollaxis(b, -2)
                b_2d = b_.reshape((b.shape[-2], -1))
                ret = a @ b_2d
                ret = ret.reshape(a.shape[0], *b_.shape[1:])
            elif sparse.issparse(b):
                # sparse is always 2D. Implies a is 3D+
                # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
                a_2d = a.reshape(-1, a.shape[-1])
                ret = a_2d @ b
                ret = ret.reshape(*a.shape[:-1], b.shape[1])
            else:
                ret = np.dot(a, b)
        else:
            ret = a @ b

        if (
            sparse.issparse(a)
            and sparse.issparse(b)
            and dense_output
            and hasattr(ret, "toarray")
        ):
            return ret.toarray()

    return ret


def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.
    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms
