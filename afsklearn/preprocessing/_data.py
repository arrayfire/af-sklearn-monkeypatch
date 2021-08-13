import warnings

import arrayfire as af
import numpy as np
from scipy import sparse, stats
from sklearn.preprocessing._data import BOUNDS_THRESHOLD
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1, inplace_csr_row_normalize_l2
from sklearn.utils.validation import FLOAT_DTYPES, _deprecate_positional_args

from .._validation import check_array, check_is_fitted, check_random_state
from .._sparsefuncs import min_max_axis
from .._extmath import row_norms
from ..base import afBaseEstimator, afTransformerMixin


def interp1_af(pos, x, y):
    # distances to normalize between two data points
    dists = af.diff1(x)

    # tile data locations to test against all desired positions
    # x = af.tile(x, 1, pos.dims()[0])
    # get indices of two nearest data points
    # idxs = af.count(x < af.tile(pos, 1, x.dims()[0]).T, dim=0).as_type(af.Dtype.s32) - 1;

    idxs = af.constant(0.0, 1, pos.dims()[0], dtype=af.Dtype.s32)
    for i in range(pos.dims()[0]):
        temp = af.tile(pos[i], x.dims()[0])
        idxs[0, i] = af.count(temp > x, 0).as_type(af.Dtype.s32) - 1

    idxs[idxs < 0] = 0
    idxs[idxs > (x.dims()[0] - 1)] = x.dims()[0] - 1

    minvals = x[idxs]

    pos -= minvals
    pos /= dists[idxs]
    pos += idxs.T

    return af.approx1(y, pos)


class QuantileTransformer(afTransformerMixin, afBaseEstimator):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    .. versionadded:: 0.19

    Parameters
    ----------
    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int, default=1e5
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see ``subsample`` for more details.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`

    copy : bool, default=True
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array).

    Attributes
    ----------
    n_quantiles_ : int
        The actual number of quantiles used to discretize the cumulative
        distribution function.

    quantiles_ : ndarray of shape (n_quantiles, n_features)
        The values corresponding the quantiles of reference.

    references_ : ndarray of shape (n_quantiles, )
        Quantiles of references.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import QuantileTransformer
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> qt = QuantileTransformer(n_quantiles=10, random_state=0)
    >>> qt.fit_transform(X)
    array([...])

    See Also
    --------
    quantile_transform : Equivalent function without the estimator API.
    PowerTransformer : Perform mapping to a normal distribution using a power
        transform.
    StandardScaler : Perform standardization that is faster, but less robust
        to outliers.
    RobustScaler : Perform robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """

    @_deprecate_positional_args
    def __init__(self, *, n_quantiles=1000, output_distribution='uniform',
                 ignore_implicit_zeros=False, subsample=int(1e5),
                 random_state=None, copy=True):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            warnings.warn("'ignore_implicit_zeros' takes effect only with"
                          " sparse matrix. This parameter has no effect.")

        n_samples, n_features = X.shape
        references = self.references_ * 100

        # TODO: fix slowdowns heeere
        self.quantiles_ = []
        for col in X.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(n_samples,
                                                    size=self.subsample,
                                                    replace=False)
                col = col.take(subsample_idx, mode='clip')
            self.quantiles_.append(np.nanpercentile(col, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        # Due to floating-point precision error in `np.nanpercentile`,
        # make sure that quantiles are monotonically increasing.
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def _sparse_fit(self, X, random_state):
        """Compute percentiles for sparse matrices.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis. The sparse matrix
            needs to be nonnegative. If a sparse matrix is provided,
            it will be converted into a sparse ``csc_matrix``.
        """

    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        y : None
            Ignored.

        Returns
        -------
        self : object
           Fitted transformer.
        """
        if self.n_quantiles <= 0:
            raise ValueError("Invalid value for 'n_quantiles': %d. "
                             "The number of quantiles must be at least one."
                             % self.n_quantiles)

        if self.subsample <= 0:
            raise ValueError("Invalid value for 'subsample': %d. "
                             "The number of subsamples must be at least one."
                             % self.subsample)

        if self.n_quantiles > self.subsample:
            raise ValueError("The number of quantiles cannot be greater than"
                             " the number of samples used. Got {} quantiles"
                             " and {} samples.".format(self.n_quantiles,
                                                       self.subsample))

        X = self._check_inputs(X, in_fit=True, copy=False)
        n_samples = X.shape[0]

        if self.n_quantiles > n_samples:
            warnings.warn("n_quantiles (%s) is greater than the total number "
                          "of samples (%s). n_quantiles is set to "
                          "n_samples."
                          % (self.n_quantiles, n_samples))
        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        rng = check_random_state(self.random_state)

        # Create the quantiles of reference
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
        if sparse.issparse(X):
            self._sparse_fit(X, rng)
        else:
            self._dense_fit(X, rng)

        return self

    def _transform_col(self, X_col, quantiles, inverse):
        """Private function to transform a single feature."""
        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    X_col = stats.norm.cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
            if output_distribution == 'normal':
                lower_bounds_idx = (X_col - BOUNDS_THRESHOLD <
                                    lower_bound_x)
                upper_bounds_idx = (X_col + BOUNDS_THRESHOLD >
                                    upper_bound_x)
            if output_distribution == 'uniform':
                lower_bounds_idx = (X_col == lower_bound_x)
                upper_bounds_idx = (X_col == upper_bound_x)

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col[isfinite_mask] = .5 * (
                np.interp(X_col_finite, quantiles, self.references_)
                - np.interp(-X_col_finite, -quantiles[::-1],
                            -self.references_[::-1]))
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite,
                                             self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    X_col = stats.norm.ppf(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD -
                                                   np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged
        return X_col

    def _transform_col_af(self, X_col, quantiles, inverse):
        """Private function to transform a single feature."""

        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    X_col = stats.norm.cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
            if output_distribution == 'normal':
                lower_bounds_idx = (X_col - BOUNDS_THRESHOLD <
                                    lower_bound_x)
                upper_bounds_idx = (X_col + BOUNDS_THRESHOLD >
                                    upper_bound_x)
            if output_distribution == 'uniform':
                lower_bounds_idx = (X_col == lower_bound_x)
                upper_bounds_idx = (X_col == upper_bound_x)

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]

        X_col_af = af.interop.from_ndarray(X_col)
        isfinite_mask_af = ~af.isnan(X_col_af)
        X_col_finite_af = X_col_af[isfinite_mask_af]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            quantiles_af = af.interop.from_ndarray(quantiles)
            references_af = af.interop.from_ndarray(self.references_)
            ires0 = np.interp(X_col_finite, quantiles, self.references_)
            # ires0_af = af.approx(X_col_finite, quantiles, self.references_)
            ires0_af = interp1_af(X_col_finite_af, quantiles_af, references_af)
            print(np.max(ires0_af.to_ndarray() - ires0))

            X_col[isfinite_mask] = .5 * (
                np.interp(X_col_finite, quantiles, self.references_)
                - np.interp(-X_col_finite, -quantiles[::-1],
                            -self.references_[::-1]))
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite,
                                             self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    X_col = stats.norm.ppf(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD -
                                                   np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged
        return X_col

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False,
                      copy=False):
        """Check inputs before fit and transform."""
        X = self._validate_data(X, reset=in_fit,
                                accept_sparse='csc', copy=copy,
                                dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')
        # we only accept positive sparse matrix when ignore_implicit_zeros is
        # false and that we call fit or transform.
        if (not accept_sparse_negative and not self.ignore_implicit_zeros
                and (sparse.issparse(X) and np.any(X.data < 0))):
            raise ValueError('QuantileTransformer only accepts'
                             ' non-negative sparse matrices.')

        # check the output distribution
        if self.output_distribution not in ('normal', 'uniform'):
            raise ValueError("'output_distribution' has to be either 'normal'"
                             " or 'uniform'. Got '{}' instead.".format(
                                 self.output_distribution))

        return X

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        if sparse.issparse(X):
            for feature_idx in range(X.shape[1]):
                column_slice = slice(X.indptr[feature_idx],
                                     X.indptr[feature_idx + 1])
                X.data[column_slice] = self._transform_col(
                    X.data[column_slice], self.quantiles_[:, feature_idx],
                    inverse)
        else:
            for feature_idx in range(X.shape[1]):
                feature_update = self._transform_col(
                    X[:, feature_idx], self.quantiles_[:, feature_idx],
                    inverse)
                # af_feature_update = self._transform_col_af(
                #     X[:, feature_idx], self.quantiles_[:, feature_idx],
                #     inverse)
                X[:, feature_idx] = feature_update
        return X

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(X, in_fit=False, copy=self.copy)

        return self._transform(X, inverse=False)


def normalize(X, norm='l2', *, axis=1, copy=True, return_norm=False):
    """Scale input vectors individually to unit norm (vector length).
    Read more in the :ref:`User Guide <preprocessing_normalization>`.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).
    axis : {0, 1}, default=1
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.
    copy : bool, default=True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).
    return_norm : bool, default=False
        whether to return the computed norms
    Returns
    -------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Normalized input X.
    norms : ndarray of shape (n_samples, ) if axis=1 else (n_features, )
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.
    See Also
    --------
    Normalizer : Performs normalization using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).
    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        sparse_format = 'csc'
    elif axis == 1:
        sparse_format = 'csr'
    else:
        raise ValueError("'%d' is not a supported axis" % axis)

    X = check_array(X, accept_sparse=sparse_format, copy=copy,
                    estimator='the normalize function', dtype=FLOAT_DTYPES)
    if axis == 0:
        X = X.T

    if sparse.issparse(X):
        if return_norm and norm in ('l1', 'l2'):
            raise NotImplementedError("return_norm=True is not implemented "
                                      "for sparse matrices with norm 'l1' "
                                      "or norm 'l2'")
        if norm == 'l1':
            inplace_csr_row_normalize_l1(X)
        elif norm == 'l2':
            inplace_csr_row_normalize_l2(X)
        elif norm == 'max':
            mins, maxes = min_max_axis(X, 1)
            norms = np.maximum(abs(mins), maxes)
            norms_elementwise = norms.repeat(np.diff(X.indptr))
            mask = norms_elementwise != 0
            X.data[mask] /= norms_elementwise[mask]
    else:
        if norm == 'l1':
            norms = np.abs(X).sum(axis=1)
        elif norm == 'l2':
            norms = row_norms(X)
        elif norm == 'max':
            norms = np.max(abs(X), axis=1)
        norms = _handle_zeros_in_scale(norms, copy=False)
        X /= norms[:, np.newaxis]

    if axis == 0:
        X = X.T

    if return_norm:
        return X, norms
    else:
        return X


def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale
