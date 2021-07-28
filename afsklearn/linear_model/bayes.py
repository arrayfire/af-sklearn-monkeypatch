from math import log

import numpy as np  # FIXME
from scipy import linalg
from sklearn.utils import _deprecate_positional_args

from .._validation import _check_sample_weight
from ..base import afRegressorMixin
from .base import _rescale_data, afLinearModel


class BayesianRidge(afRegressorMixin, afLinearModel):
    """Bayesian ridge regression.
    Fit a Bayesian ridge model. See the Notes section for details on this
    implementation and the optimization of the regularization parameters
    lambda (precision of the weights) and alpha (precision of the noise).
    Read more in the :ref:`User Guide <bayesian_regression>`.
    Parameters
    ----------
    n_iter : int, default=300
        Maximum number of iterations. Should be greater than or equal to 1.
    tol : float, default=1e-3
        Stop the algorithm if w has converged.
    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.
    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.
    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.
    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.
    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
        If not set, alpha_init is 1/Var(y).
            .. versionadded:: 0.22
    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
        If not set, lambda_init is 1.
            .. versionadded:: 0.22
    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        The intercept is not treated as a probabilistic parameter
        and thus has no associated variance. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    verbose : bool, default=False
        Verbose mode when fitting the model.
    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)
    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.
    alpha_ : float
       Estimated precision of the noise.
    lambda_ : float
       Estimated precision of the weights.
    sigma_ : array-like of shape (n_features, n_features)
        Estimated variance-covariance matrix of the weights
    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood (to be
        maximized) at each iteration of the optimization. The array starts
        with the value of the log marginal likelihood obtained for the initial
        values of alpha and lambda and ends with the value obtained for the
        estimated alpha and lambda.
    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
    X_offset_ : float
        If `normalize=True`, offset subtracted for centering data to a
        zero mean.
    X_scale_ : float
        If `normalize=True`, parameter used to scale data to a unit
        standard deviation.
    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.BayesianRidge()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    BayesianRidge()
    >>> clf.predict([[1, 1]])
    array([1.])
    Notes
    -----
    There exist several strategies to perform Bayesian ridge regression. This
    implementation is based on the algorithm described in Appendix A of
    (Tipping, 2001) where updates of the regularization parameters are done as
    suggested in (MacKay, 1992). Note that according to A New
    View of Automatic Relevance Determination (Wipf and Nagarajan, 2008) these
    update rules do not guarantee that the marginal likelihood is increasing
    between two consecutive iterations of the optimization.
    References
    ----------
    D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
    Vol. 4, No. 3, 1992.
    M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
    Journal of Machine Learning Research, Vol. 1, 2001.
    """
    @_deprecate_positional_args
    def __init__(self, *, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6,
                 lambda_1=1.e-6, lambda_2=1.e-6, alpha_init=None,
                 lambda_init=None, compute_score=False, fit_intercept=True,
                 normalize=False, copy_X=True, verbose=False):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the model
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary
        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample
            .. versionadded:: 0.20
               parameter *sample_weight* support to BayesianRidge.
        Returns
        -------
        self : returns an instance of self.
        """

        if self.n_iter < 1:
            raise ValueError('n_iter should be greater than or equal to 1.'
                             ' Got {!r}.'.format(self.n_iter))

        X, y = self._validate_data(X, y, dtype=np.float64, y_numeric=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        X, y, X_offset_, y_offset_, X_scale_ = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_
        n_samples, n_features = X.shape

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero
        alpha_ = self.alpha_init
        lambda_ = self.lambda_init
        if alpha_ is None:
            alpha_ = 1. / (np.var(y) + eps)
        if lambda_ is None:
            lambda_ = 1.

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S ** 2

        # Convergence loop of the bayesian ridge regression
        for iter_ in range(self.n_iter):

            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding rmse
            coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                              XT_y, U, Vh, eigen_vals_,
                                              alpha_, lambda_)
            if self.compute_score:
                # compute the log marginal likelihood
                s = self._log_marginal_likelihood(n_samples, n_features,
                                                  eigen_vals_,
                                                  alpha_, lambda_,
                                                  coef_, rmse_)
                self.scores_.append(s)

            # Update alpha and lambda according to (MacKay, 1992)
            gamma_ = np.sum((alpha_ * eigen_vals_) /
                            (lambda_ + alpha_ * eigen_vals_))
            lambda_ = ((gamma_ + 2 * lambda_1) /
                       (np.sum(coef_ ** 2) + 2 * lambda_2))
            alpha_ = ((n_samples - gamma_ + 2 * alpha_1) /
                      (rmse_ + 2 * alpha_2))

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                               XT_y, U, Vh, eigen_vals_,
                                               alpha_, lambda_)
        if self.compute_score:
            # compute the log marginal likelihood
            s = self._log_marginal_likelihood(n_samples, n_features,
                                              eigen_vals_,
                                              alpha_, lambda_,
                                              coef_, rmse_)
            self.scores_.append(s)
            self.scores_ = np.array(self.scores_)

        # posterior covariance is given by 1/alpha_ * scaled_sigma_
        scaled_sigma_ = np.dot(Vh.T,
                               Vh / (eigen_vals_ +
                                     lambda_ / alpha_)[:, np.newaxis])
        self.sigma_ = (1. / alpha_) * scaled_sigma_

        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self

    def predict(self, X, return_std=False):
        """Predict using the linear model.
        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.
        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.
        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.
        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        y_mean = self._decision_function(X)
        if return_std is False:
            return y_mean
        else:
            if self.normalize:
                X = (X - self.X_offset_) / self.X_scale_
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1. / self.alpha_))
            return y_mean, y_std

    def _update_coef_(self, X, y, n_samples, n_features, XT_y, U, Vh,
                      eigen_vals_, alpha_, lambda_):
        """Update posterior mean and compute corresponding rmse.
        Posterior mean is given by coef_ = scaled_sigma_ * X.T * y where
        scaled_sigma_ = (lambda_/alpha_ * np.eye(n_features)
                         + np.dot(X.T, X))^-1
        """

        if n_samples > n_features:
            coef_ = np.linalg.multi_dot([Vh.T,
                                         Vh / (eigen_vals_ + lambda_ /
                                               alpha_)[:, np.newaxis],
                                         XT_y])
        else:
            coef_ = np.linalg.multi_dot([X.T,
                                         U / (eigen_vals_ + lambda_ /
                                              alpha_)[None, :],
                                         U.T, y])

        rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)

        return coef_, rmse_

    def _log_marginal_likelihood(self, n_samples, n_features, eigen_vals,
                                 alpha_, lambda_, coef, rmse):
        """Log marginal likelihood."""
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # compute the log of the determinant of the posterior covariance.
        # posterior covariance is given by
        # sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
        if n_samples > n_features:
            logdet_sigma = - np.sum(np.log(lambda_ + alpha_ * eigen_vals))
        else:
            logdet_sigma = np.full(n_features, lambda_,
                                   dtype=np.array(lambda_).dtype)
            logdet_sigma[:n_samples] += alpha_ * eigen_vals
            logdet_sigma = - np.sum(np.log(logdet_sigma))

        score = lambda_1 * log(lambda_) - lambda_2 * lambda_
        score += alpha_1 * log(alpha_) - alpha_2 * alpha_
        score += 0.5 * (n_features * log(lambda_) +
                        n_samples * log(alpha_) -
                        alpha_ * rmse -
                        lambda_ * np.sum(coef ** 2) +
                        logdet_sigma -
                        n_samples * log(2 * np.pi))

        return score
