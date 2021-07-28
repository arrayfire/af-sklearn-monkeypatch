import warnings
from abc import ABCMeta, abstractmethod

import numpy as np  # FIXME
from joblib import Parallel
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import SparseCoefMixin
from sklearn.linear_model._sgd_fast import (  # FIXME
    EpsilonInsensitive, Hinge, Huber, Log, ModifiedHuber, SquaredEpsilonInsensitive, SquaredHinge, SquaredLoss,
    _plain_sgd)
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON, LEARNING_RATE_TYPES, MAX_INT, PENALTY_TYPES
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.utils import deprecated
from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.utils.validation import _deprecate_positional_args

from .._class_weight import compute_class_weight
from .._classifier_mixin import afLinearClassifierMixin
from .._multiclass import _check_partial_fit_first_call
from .._validation import _check_sample_weight, check_random_state, check_X_y
from ..base import afBaseEstimator
from .base import make_dataset


class _ValidationScoreCallback:
    """Callback for early stopping based on validation score"""

    def __init__(self, estimator, X_val, y_val, sample_weight_val,
                 classes=None):
        self.estimator = clone(estimator)
        self.estimator.t_ = 1  # to pass check_is_fitted
        if classes is not None:
            self.estimator.classes_ = classes
        self.X_val = X_val
        self.y_val = y_val
        self.sample_weight_val = sample_weight_val

    def __call__(self, coef, intercept):
        est = self.estimator
        est.coef_ = coef.reshape(1, -1)
        est.intercept_ = np.atleast_1d(intercept)
        return est.score(self.X_val, self.y_val, self.sample_weight_val)


class afBaseSGD(SparseCoefMixin, afBaseEstimator, metaclass=ABCMeta):
    """Base class for SGD classification and regression."""
    @_deprecate_positional_args
    def __init__(self, loss, *, penalty='l2', alpha=0.0001, C=1.0,
                 l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
                 shuffle=True, verbose=0, epsilon=0.1, random_state=None,
                 learning_rate="optimal", eta0=0.0, power_t=0.5,
                 early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, warm_start=False, average=False):
        self.loss = loss
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.alpha = alpha
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average
        self.max_iter = max_iter
        self.tol = tol
        # current tests expect init to do parameter validation
        # but we are not allowed to set attributes
        self._validate_params()

    def set_params(self, **kwargs):
        """Set and validate the parameters of estimator.
        Parameters
        ----------
        **kwargs : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        super().set_params(**kwargs)
        self._validate_params()
        return self

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _validate_params(self, for_partial_fit=False):
        """Validate input params. """
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False")
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False")
        if self.early_stopping and for_partial_fit:
            raise ValueError("early_stopping should be False with partial_fit")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be > zero. Got %f" % self.max_iter)
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if self.n_iter_no_change < 1:
            raise ValueError("n_iter_no_change must be >= 1")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in range (0, 1)")
        if self.learning_rate in ("constant", "invscaling", "adaptive"):
            if self.eta0 <= 0.0:
                raise ValueError("eta0 must be > 0")
        if self.learning_rate == "optimal" and self.alpha == 0:
            raise ValueError("alpha must be > 0 since "
                             "learning_rate is 'optimal'. alpha is used "
                             "to compute the optimal learning rate.")

        # raises ValueError if not registered
        self._get_penalty_type(self.penalty)
        self._get_learning_rate_type(self.learning_rate)

        if self.loss not in self.loss_functions:
            raise ValueError("The loss %s is not supported. " % self.loss)

    def _get_loss_function(self, loss):
        """Get concrete ``LossFunction`` object for str ``loss``. """
        try:
            loss_ = self.loss_functions[loss]
            loss_class, args = loss_[0], loss_[1:]
            if loss in ('huber', 'epsilon_insensitive',
                        'squared_epsilon_insensitive'):
                args = (self.epsilon, )
            return loss_class(*args)
        except KeyError as e:
            raise ValueError("The loss %s is not supported. " % loss) from e

    def _get_learning_rate_type(self, learning_rate):
        try:
            return LEARNING_RATE_TYPES[learning_rate]
        except KeyError as e:
            raise ValueError("learning rate %s "
                             "is not supported. " % learning_rate) from e

    def _get_penalty_type(self, penalty):
        penalty = str(penalty).lower()
        try:
            return PENALTY_TYPES[penalty]
        except KeyError as e:
            raise ValueError("Penalty %s is not supported. " % penalty) from e

    def _allocate_parameter_mem(self, n_classes, n_features, coef_init=None,
                                intercept_init=None):
        """Allocate mem for parameters; initialize if provided."""
        if n_classes > 2:
            # allocate coef_ for multi-class
            if coef_init is not None:
                coef_init = np.asarray(coef_init, order="C")
                if coef_init.shape != (n_classes, n_features):
                    raise ValueError("Provided ``coef_`` does not match "
                                     "dataset. ")
                self.coef_ = coef_init
            else:
                self.coef_ = np.zeros((n_classes, n_features),
                                      dtype=np.float64, order="C")

            # allocate intercept_ for multi-class
            if intercept_init is not None:
                intercept_init = np.asarray(intercept_init, order="C")
                if intercept_init.shape != (n_classes, ):
                    raise ValueError("Provided intercept_init "
                                     "does not match dataset.")
                self.intercept_ = intercept_init
            else:
                self.intercept_ = np.zeros(n_classes, dtype=np.float64,
                                           order="C")
        else:
            # allocate coef_ for binary problem
            if coef_init is not None:
                coef_init = np.asarray(coef_init, dtype=np.float64,
                                       order="C")
                coef_init = coef_init.ravel()
                if coef_init.shape != (n_features,):
                    raise ValueError("Provided coef_init does not "
                                     "match dataset.")
                self.coef_ = coef_init
            else:
                self.coef_ = np.zeros(n_features,
                                      dtype=np.float64,
                                      order="C")

            # allocate intercept_ for binary problem
            if intercept_init is not None:
                intercept_init = np.asarray(intercept_init, dtype=np.float64)
                if intercept_init.shape != (1,) and intercept_init.shape != ():
                    raise ValueError("Provided intercept_init "
                                     "does not match dataset.")
                self.intercept_ = intercept_init.reshape(1,)
            else:
                self.intercept_ = np.zeros(1, dtype=np.float64, order="C")

        # initialize average parameters
        if self.average > 0:
            self._standard_coef = self.coef_
            self._standard_intercept = self.intercept_
            self._average_coef = np.zeros(self.coef_.shape,
                                          dtype=np.float64,
                                          order="C")
            self._average_intercept = np.zeros(self._standard_intercept.shape,
                                               dtype=np.float64,
                                               order="C")

    def _make_validation_split(self, y):
        """Split the dataset between training set and validation set.
        Parameters
        ----------
        y : ndarray of shape (n_samples, )
            Target values.
        Returns
        -------
        validation_mask : ndarray of shape (n_samples, )
            Equal to 1 on the validation set, 0 on the training set.
        """
        n_samples = y.shape[0]
        validation_mask = np.zeros(n_samples, dtype=np.uint8)
        if not self.early_stopping:
            # use the full set for training, with an empty validation set
            return validation_mask

        if is_classifier(self):
            splitter_type = StratifiedShuffleSplit
        else:
            splitter_type = ShuffleSplit
        cv = splitter_type(test_size=self.validation_fraction,
                           random_state=self.random_state)
        idx_train, idx_val = next(cv.split(np.zeros(shape=(y.shape[0], 1)), y))
        if idx_train.shape[0] == 0 or idx_val.shape[0] == 0:
            raise ValueError(
                "Splitting %d samples into a train set and a validation set "
                "with validation_fraction=%r led to an empty set (%d and %d "
                "samples). Please either change validation_fraction, increase "
                "number of samples, or disable early_stopping."
                % (n_samples, self.validation_fraction, idx_train.shape[0],
                   idx_val.shape[0]))

        validation_mask[idx_val] = 1
        return validation_mask

    def _make_validation_score_cb(self, validation_mask, X, y, sample_weight,
                                  classes=None):
        if not self.early_stopping:
            return None

        return _ValidationScoreCallback(
            self, X[validation_mask], y[validation_mask],
            sample_weight[validation_mask], classes=classes)

    # mypy error: Decorated property not supported
    @deprecated("Attribute standard_coef_ was deprecated "  # type: ignore
                "in version 0.23 and will be removed in 1.0 "
                "(renaming of 0.25).")
    @property
    def standard_coef_(self):
        return self._standard_coef

    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute standard_intercept_ was deprecated "
        "in version 0.23 and will be removed in 1.0 (renaming of 0.25)."
    )
    @property
    def standard_intercept_(self):
        return self._standard_intercept

    # mypy error: Decorated property not supported
    @deprecated("Attribute average_coef_ was deprecated "  # type: ignore
                "in version 0.23 and will be removed in 1.0 "
                "(renaming of 0.25).")
    @property
    def average_coef_(self):
        return self._average_coef

    # mypy error: Decorated property not supported
    @deprecated("Attribute average_intercept_ was deprecated "  # type: ignore
                "in version 0.23 and will be removed in 1.0 "
                "(renaming of 0.25).")
    @property
    def average_intercept_(self):
        return self._average_intercept


class afBaseSGDClassifier(afLinearClassifierMixin, afBaseSGD, metaclass=ABCMeta):

    loss_functions = {
        "hinge": (Hinge, 1.0),
        "squared_hinge": (SquaredHinge, 1.0),
        "perceptron": (Hinge, 0.0),
        "log": (Log, ),
        "modified_huber": (ModifiedHuber, ),
        "squared_loss": (SquaredLoss, ),
        "huber": (Huber, DEFAULT_EPSILON),
        "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        "squared_epsilon_insensitive": (SquaredEpsilonInsensitive,
                                        DEFAULT_EPSILON),
    }

    @abstractmethod
    @_deprecate_positional_args
    def __init__(self, loss="hinge", *, penalty='l2', alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
                 shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=None,
                 random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5,
                 class_weight=None, warm_start=False, average=False):

        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, warm_start=warm_start,
            average=average)
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def _partial_fit(self, X, y, alpha, C,
                     loss, learning_rate, max_iter,
                     classes, sample_weight,
                     coef_init, intercept_init):
        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                         order="C", accept_large_sparse=False)

        n_samples, n_features = X.shape

        _check_partial_fit_first_call(self, classes)

        n_classes = self.classes_.shape[0]

        # Allocate datastructures from input arguments
        self._expanded_class_weight = compute_class_weight(
            self.class_weight, classes=self.classes_, y=y)
        sample_weight = _check_sample_weight(sample_weight, X)

        if getattr(self, "coef_", None) is None or coef_init is not None:
            self._allocate_parameter_mem(n_classes, n_features,
                                         coef_init, intercept_init)
        elif n_features != self.coef_.shape[-1]:
            raise ValueError("Number of features %d does not match previous "
                             "data %d." % (n_features, self.coef_.shape[-1]))

        self.loss_function_ = self._get_loss_function(loss)
        if not hasattr(self, "t_"):
            self.t_ = 1.0

        # delegate to concrete training procedure
        if n_classes > 2:
            self._fit_multiclass(X, y, alpha=alpha, C=C,
                                 learning_rate=learning_rate,
                                 sample_weight=sample_weight,
                                 max_iter=max_iter)
        elif n_classes == 2:
            self._fit_binary(X, y, alpha=alpha, C=C,
                             learning_rate=learning_rate,
                             sample_weight=sample_weight,
                             max_iter=max_iter)
        else:
            raise ValueError(
                "The number of classes has to be greater than one;"
                " got %d class" % n_classes)

        return self

    def _fit(self, X, y, alpha, C, loss, learning_rate, coef_init=None,
             intercept_init=None, sample_weight=None):
        self._validate_params()
        if hasattr(self, "classes_"):
            self.classes_ = None

        X, y = self._validate_data(X, y, accept_sparse='csr',
                                   dtype=np.float64, order="C",
                                   accept_large_sparse=False)

        # labels can be encoded as float, int, or string literals
        # np.unique sorts in asc order; largest class id is positive class
        classes = np.unique(y)

        if self.warm_start and hasattr(self, "coef_"):
            if coef_init is None:
                coef_init = self.coef_
            if intercept_init is None:
                intercept_init = self.intercept_
        else:
            self.coef_ = None
            self.intercept_ = None

        if self.average > 0:
            self._standard_coef = self.coef_
            self._standard_intercept = self.intercept_
            self._average_coef = None
            self._average_intercept = None

        # Clear iteration count for multiple call to fit.
        self.t_ = 1.0

        self._partial_fit(X, y, alpha, C, loss, learning_rate, self.max_iter,
                          classes, sample_weight, coef_init, intercept_init)

        if (self.tol is not None and self.tol > -np.inf
                and self.n_iter_ == self.max_iter):
            warnings.warn("Maximum number of iteration reached before "
                          "convergence. Consider increasing max_iter to "
                          "improve the fit.",
                          ConvergenceWarning)
        return self

    def _fit_binary(self, X, y, alpha, C, sample_weight,
                    learning_rate, max_iter):
        """Fit a binary classifier on X and y. """
        coef, intercept, n_iter_ = fit_binary(self, 1, X, y, alpha, C,
                                              learning_rate, max_iter,
                                              self._expanded_class_weight[1],
                                              self._expanded_class_weight[0],
                                              sample_weight,
                                              random_state=self.random_state)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        # need to be 2d
        if self.average > 0:
            if self.average <= self.t_ - 1:
                self.coef_ = self._average_coef.reshape(1, -1)
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef.reshape(1, -1)
                self._standard_intercept = np.atleast_1d(intercept)
                self.intercept_ = self._standard_intercept
        else:
            self.coef_ = coef.reshape(1, -1)
            # intercept is a float, need to convert it to an array of length 1
            self.intercept_ = np.atleast_1d(intercept)

    def _fit_multiclass(self, X, y, alpha, C, learning_rate,
                        sample_weight, max_iter):
        """Fit a multi-class classifier by combining binary classifiers
        Each binary classifier predicts one class versus all others. This
        strategy is called OvA (One versus All) or OvR (One versus Rest).
        """
        # Precompute the validation split using the multiclass labels
        # to ensure proper balancing of the classes.
        validation_mask = self._make_validation_split(y)

        # Use joblib to fit OvA in parallel.
        # Pick the random seed for each job outside of fit_binary to avoid
        # sharing the estimator random state between threads which could lead
        # to non-deterministic behavior
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=len(self.classes_))
        result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                          **_joblib_parallel_args(require="sharedmem"))(
            delayed(fit_binary)(self, i, X, y, alpha, C, learning_rate,
                                max_iter, self._expanded_class_weight[i],
                                1., sample_weight,
                                validation_mask=validation_mask,
                                random_state=seed)
            for i, seed in enumerate(seeds))

        # take the maximum of n_iter_ over every binary fit
        n_iter_ = 0.
        for i, (_, intercept, n_iter_i) in enumerate(result):
            self.intercept_[i] = intercept
            n_iter_ = max(n_iter_, n_iter_i)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self._average_coef
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef
                self._standard_intercept = np.atleast_1d(self.intercept_)
                self.intercept_ = self._standard_intercept

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Perform one epoch of stochastic gradient descent on given samples.
        Internally, this method uses ``max_iter = 1``. Therefore, it is not
        guaranteed that a minimum of the cost function is reached after calling
        it once. Matters such as objective convergence and early stopping
        should be handled by the user.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data.
        y : ndarray of shape (n_samples,)
            Subset of the target values.
        classes : ndarray of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.
        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.
        Returns
        -------
        self :
            Returns an instance of self.
        """
        self._validate_params(for_partial_fit=True)
        if self.class_weight in ['balanced']:
            raise ValueError("class_weight '{0}' is not supported for "
                             "partial_fit. In order to use 'balanced' weights,"
                             " use compute_class_weight('{0}', "
                             "classes=classes, y=y). "
                             "In place of y you can us a large enough sample "
                             "of the full training set target to properly "
                             "estimate the class frequency distributions. "
                             "Pass the resulting weights as the class_weight "
                             "parameter.".format(self.class_weight))
        return self._partial_fit(X, y, alpha=self.alpha, C=1.0, loss=self.loss,
                                 learning_rate=self.learning_rate, max_iter=1,
                                 classes=classes, sample_weight=sample_weight,
                                 coef_init=None, intercept_init=None)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        coef_init : ndarray of shape (n_classes, n_features), default=None
            The initial coefficients to warm-start the optimization.
        intercept_init : ndarray of shape (n_classes,), default=None
            The initial intercept to warm-start the optimization.
        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed. These weights will
            be multiplied with class_weight (passed through the
            constructor) if class_weight is specified.
        Returns
        -------
        self :
            Returns an instance of self.
        """
        return self._fit(X, y, alpha=self.alpha, C=1.0,
                         loss=self.loss, learning_rate=self.learning_rate,
                         coef_init=coef_init, intercept_init=intercept_init,
                         sample_weight=sample_weight)


def fit_binary(est, i, X, y, alpha, C, learning_rate, max_iter,
               pos_weight, neg_weight, sample_weight, validation_mask=None,
               random_state=None):
    """Fit a single binary classifier.
    The i'th class is considered the "positive" class.
    Parameters
    ----------
    est : Estimator object
        The estimator to fit
    i : int
        Index of the positive class
    X : numpy array or sparse matrix of shape [n_samples,n_features]
        Training data
    y : numpy array of shape [n_samples, ]
        Target values
    alpha : float
        The regularization parameter
    C : float
        Maximum step size for passive aggressive
    learning_rate : string
        The learning rate. Accepted values are 'constant', 'optimal',
        'invscaling', 'pa1' and 'pa2'.
    max_iter : int
        The maximum number of iterations (epochs)
    pos_weight : float
        The weight of the positive class
    neg_weight : float
        The weight of the negative class
    sample_weight : numpy array of shape [n_samples, ]
        The weight of each sample
    validation_mask : numpy array of shape [n_samples, ], default=None
        Precomputed validation mask in case _fit_binary is called in the
        context of a one-vs-rest reduction.
    random_state : int, RandomState instance, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    # if average is not true, average_coef, and average_intercept will be
    # unused
    y_i, coef, intercept, average_coef, average_intercept = \
        _prepare_fit_binary(est, y, i)
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]

    random_state = check_random_state(random_state)
    dataset, intercept_decay = make_dataset(
        X, y_i, sample_weight, random_state=random_state)

    penalty_type = est._get_penalty_type(est.penalty)
    learning_rate_type = est._get_learning_rate_type(learning_rate)

    if validation_mask is None:
        validation_mask = est._make_validation_split(y_i)
    classes = np.array([-1, 1], dtype=y_i.dtype)
    validation_score_cb = est._make_validation_score_cb(
        validation_mask, X, y_i, sample_weight, classes=classes)

    # numpy mtrand expects a C long which is a signed 32 bit integer under
    # Windows
    seed = random_state.randint(MAX_INT)

    tol = est.tol if est.tol is not None else -np.inf

    coef, intercept, average_coef, average_intercept, n_iter_ = _plain_sgd(
        coef, intercept, average_coef, average_intercept, est.loss_function_,
        penalty_type, alpha, C, est.l1_ratio, dataset, validation_mask,
        est.early_stopping, validation_score_cb, int(est.n_iter_no_change),
        max_iter, tol, int(est.fit_intercept), int(est.verbose),
        int(est.shuffle), seed, pos_weight, neg_weight, learning_rate_type,
        est.eta0, est.power_t, est.t_, intercept_decay, est.average)

    if est.average:
        if len(est.classes_) == 2:
            est._average_intercept[0] = average_intercept
        else:
            est._average_intercept[i] = average_intercept

    return coef, intercept, n_iter_


def _prepare_fit_binary(est, y, i):
    """Initialization for fit_binary.
    Returns y, coef, intercept, average_coef, average_intercept.
    """
    y_i = np.ones(y.shape, dtype=np.float64, order="C")
    y_i[y != est.classes_[i]] = -1.0
    average_intercept = 0
    average_coef = None

    if len(est.classes_) == 2:
        if not est.average:
            coef = est.coef_.ravel()
            intercept = est.intercept_[0]
        else:
            coef = est._standard_coef.ravel()
            intercept = est._standard_intercept[0]
            average_coef = est._average_coef.ravel()
            average_intercept = est._average_intercept[0]
    else:
        if not est.average:
            coef = est.coef_[i]
            intercept = est.intercept_[i]
        else:
            coef = est._standard_coef[i]
            intercept = est._standard_intercept[i]
            average_coef = est._average_coef[i]
            average_intercept = est._average_intercept[i]

    return y_i, coef, intercept, average_coef, average_intercept
