import warnings
from abc import ABCMeta, abstractmethod
from math import sqrt

import arrayfire as af
import numpy as np
import scipy
import scipy.sparse
from sklearn.base import is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state, gen_batches, shuffle
from sklearn.utils.optimize import _check_optimize_result

from .._extmath import safe_sparse_dot
from .._nn_utils import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from .._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from .._validation import _safe_indexing, check_array
from ..base import afBaseEstimator
import time

_STOCHASTIC_SOLVERS = ['sgd', 'adam']

def _pack(coefs_, intercepts_):
    """Pack the parameters into a single vector."""
    if isinstance(coefs_, af.Array): #TODO?
        pass
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])

def af_type_matmulTN(a, b):
    if a.type() == af.Dtype.f64 or b.type() == af.Dtype.f64:
        ret = af.blas.matmulTN(a.as_type(af.Dtype.f64), b.as_type(af.Dtype.f64))
    else:
        ret = af.blas.matmulTN(a.as_type(af.Dtype.f32), b.as_type(af.Dtype.f32))
    return ret

def cvtArgsaToAf(X, y, activations, deltas, coef_grads, intercept_grads):
    if scipy.sparse.issparse(X):
        X = X.todense()
    if activations and scipy.sparse.issparse(activations[0]):
        activations = [a.todense() if a is not None else a for a in activations]

    X_af = af.interop.from_ndarray(X)
    y_af = af.interop.from_ndarray(y)

    if activations and not isinstance(activations[0], af.Array):
        activations_af = [af.interop.from_ndarray(a)  if a is not None else a for a in activations]
    else:
        activations_af = activations

    if deltas and not isinstance(deltas[0], af.Array):
        deltas_af = [af.interop.from_ndarray(a)  if a is not None else a for a in deltas]
    else:
        deltas_af = deltas

    if coef_grads and not isinstance(coef_grads[0], af.Array):
        coef_grads_af = [af.interop.from_ndarray(a)  if a is not None else a for a in coef_grads]
    else:
        coef_grads_af = coef_grads

    if intercept_grads and not isinstance(intercept_grads[0], af.Array):
        intercept_grads_af = [af.interop.from_ndarray(a)  if a is not None else a for a in intercept_grads]
    else:
        intercept_grads_af = intercept_grads

    return X_af, y_af, activations_af, deltas_af, coef_grads_af, intercept_grads_af


class BaseMultilayerPerceptron(afBaseEstimator, metaclass=ABCMeta):
    """
    Base class for MLP classification and regression.
    """

    @abstractmethod
    def __init__(self, hidden_layer_sizes, activation, solver,
                 alpha, batch_size, learning_rate, learning_rate_init, power_t,
                 max_iter, loss, shuffle, random_state, tol, verbose,
                 warm_start, momentum, nesterovs_momentum, early_stopping,
                 validation_fraction, beta_1, beta_2, epsilon,
                 n_iter_no_change, max_fun):
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.loss = loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

    def _unpack(self, packed_parameters):
        """Extract the coefficients and intercepts from packed_parameters."""
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)

            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]

    def _forward_pass_af(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers

        if activations and not isinstance(activations[0], af.Array):
            activations_af = [af.interop.from_ndarray(a)  if isinstance(a, np.ndarray) else a for a in activations]
        else:
            activations_af = activations

        if self.coefs_ and not isinstance(self.coefs_[0], af.Array):
            coefs_af = [af.interop.from_ndarray(c)  if isinstance(c, np.ndarray) else c for c in self.coefs_]
            self.coefs_ = coefs_af
        else:
            coefs_af = self.coefs_

        if self.intercepts_ and not isinstance(self.intercepts_[0], af.Array):
            intercepts_af = [af.interop.from_ndarray(i)  if isinstance(i, np.ndarray) else i for i in self.intercepts_]
            self.intercepts_ = intercepts_af
        else:
            intercepts_af = self.intercepts_


        for i in range(self.n_layers_ - 1):
            #activations_af[i + 1]  = af.matmul(activations_af[i], coefs_af[i])
            activations_af[i + 1] = safe_sparse_dot(activations_af[i],
                                                    coefs_af[i])
            #activations_af[i + 1] += af.tile(intercepts_af[i], 1, activations_af[i+1].shape[0]).T
            activations_af[i + 1] += af.tile(intercepts_af[i].T, activations_af[i+1].shape[0])

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations_af[i + 1] = hidden_activation(activations_af[i + 1])

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        activations_af[i + 1] = output_activation(activations_af[i + 1])


        return activations_af

    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.
        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """

        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1]  = safe_sparse_dot(activations[i],
                                                  self.coefs_[i])
            activations[i + 1] += af.tile(self.intercepts_[i].T, activations[i+1].shape[0])

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = hidden_activation(activations[i + 1])
                #activations[i + 1] = af.tile(self.intercepts_[i].T, activations[i+1].dims()[0])



        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        activations[i + 1] = output_activation(activations[i + 1])


        return activations

    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.
        This function does backpropagation for the specified one layer.
        """
        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self.alpha * self.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = af.flat(af.mean(deltas[layer], dim=0))

        return coef_grads, intercept_grads

    def _compute_loss_grad_af(self, layer, n_samples, activations, deltas,
                              coef_grads, intercept_grads):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.

        This function does backpropagation for the specified one layer.
        """
        coef_grads[layer] = af_type_matmulTN(activations[layer], deltas[layer])
        coef_grads[layer] += (self.alpha * self.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = af.flat(af.mean(deltas[layer], dim=0))
        return coef_grads, intercept_grads

    def _loss_grad_lbfgs(self, packed_coef_inter, X, y, activations, deltas,
                         coef_grads, intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to the different parameters given in the initialization.
        Returned gradients are packed in a single vector so it can be used
        in lbfgs
        Parameters
        ----------
        packed_coef_inter : ndarray
            A vector comprising the flattened coefficients and intercepts.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,)
            The target values.
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        Returns
        -------
        loss : float
        grad : array-like, shape (number of nodes of all layers,)
        """
        self._unpack(packed_coef_inter)
        #naive conversion, TODO: avoid repeated mem-transfers
        X_af, y_af, activations_af, deltas_af, coef_grads_af, intercept_grads_af =\
        cvtArgsaToAf(X, y, activations, deltas, coef_grads, intercept_grads)
        if self.coefs_ and not isinstance(self.coefs_[0], af.Array):
            self.coefs_ = [af.interop.from_ndarray(c)  if c is not None else c for c in self.coefs_]
        if self.intercepts_ and not isinstance(self.intercepts_[0], af.Array):
            self.intercepts_ = [af.interop.from_ndarray(i)  if i is not None else i for i in self.intercepts_]

        loss, coef_grads, intercept_grads = self._backprop(
            X_af, y_af, activations_af, deltas_af, coef_grads_af, intercept_grads_af)
            #X, y, activations, deltas, coef_grads, intercept_grads)

        #conversion back
        coef_grads = [a.to_ndarray() if a is not None else a for a in coef_grads]
        intercept_grads = [np.squeeze(a.to_ndarray()) if a is not None else a for a in intercept_grads]

        grad = _pack(coef_grads, intercept_grads)
        return loss, grad

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        #y_af = af.interop.from_ndarray(y).as_type(af.Dtype.s32) if \
                        #not isinstance(y,af.Array) else y
        y_af = af.interop.from_ndarray(y).as_type(af.Dtype.f32) if \
                        not isinstance(y,af.Array) else y.as_type(af.Dtype.f32)

        # Forward propagate
        # will convert grads(coefs_ + intercepts_) to af.Array if needed
        activations = self._forward_pass_af(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y_af.as_type(af.Dtype.f32), activations[-1])

        # Add L2 regularization term to loss
        values = np.sum(np.array([af.dot(af.flat(s), af.flat(s)).scalar() for s in self.coefs_]))
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss

        deltas[last] = activations[-1] - y_af.as_type(af.Dtype.f32)

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_loss_grad_af(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)


        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            #deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            deltas[i - 1] = af.matmulNT(deltas[i], self.coefs_[i])
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_loss_grad_af(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return loss, coef_grads, intercept_grads

    def _initialize(self, y, layer_units):
        # set all attributes, allocate weights etc for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Output for regression
        if not is_classifier(self):
            self.out_activation_ = 'identity'
        # Output for multi class
        elif self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation_ = 'softmax'
        # Output for binary class and multi-label
        else:
            self.out_activation_ = 'logistic'

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            if self.solver == 'lbfgs':
                coef_init, intercept_init = self._init_coef(layer_units[i],
                                                            layer_units[i + 1], make_af_array=False)
            else:
                coef_init, intercept_init = self._init_coef(layer_units[i],
                                                            layer_units[i + 1], make_af_array=True)
                                                            #layer_units[i + 1], make_af_array=False)
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
            else:
                self.best_loss_ = np.inf

    def _init_coef(self, fan_in, fan_out, make_af_array=False):
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.
        if self.activation == 'logistic':
            factor = 2.
        init_bound = sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        if make_af_array:
            coef_init = (2 * init_bound) * af.randu(fan_in, fan_out) - init_bound
            intercept_init = (2 * init_bound) * af.randu(fan_out) - init_bound
        else:
            # Generate weights and bias:
            coef_init = self._random_state.uniform(-init_bound, init_bound,
                                                  (fan_in, fan_out))
            intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                        fan_out)

        return coef_init, intercept_init
        #return coef_init.as_type(af.Dtype.f64), intercept_init.as_type(af.Dtype.f64)

    def _fit(self, X, y, incremental=False):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                             hidden_layer_sizes)

        X, y = self._validate_input(X, y, incremental)
        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])

        # check random state
        self._random_state = check_random_state(self.random_state)

        if not hasattr(self, 'coefs_') or (not self.warm_start and not
                                           incremental):
            # First time training the model
            self._initialize(y, layer_units)

        # lbfgs does not support mini-batches
        if self.solver == 'lbfgs':
            batch_size = n_samples
        elif self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn("Got `batch_size` less than 1 or larger than "
                              "sample size. It is going to be clipped")
            batch_size = np.clip(self.batch_size, 1, n_samples)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                      n_fan_out_ in zip(layer_units[:-1],
                                        layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(X, y, activations, deltas, coef_grads,
                                 intercept_grads, layer_units, incremental)

        # Run the LBFGS solver
        elif self.solver == 'lbfgs':
            self._fit_lbfgs(X, y, activations, deltas, coef_grads,
                            intercept_grads, layer_units)
        return self


    def _validate_hyperparameters(self):
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False, got %s." %
                             self.shuffle)
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_iter)
        if self.max_fun <= 0:
            raise ValueError("max_fun must be > 0, got %s." % self.max_fun)
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0, got %s." % self.alpha)
        if (self.learning_rate in ["constant", "invscaling", "adaptive"] and
                self.learning_rate_init <= 0.0):
            raise ValueError("learning_rate_init must be > 0, got %s." %
                             self.learning_rate)
        if self.momentum > 1 or self.momentum < 0:
            raise ValueError("momentum must be >= 0 and <= 1, got %s" %
                             self.momentum)
        if not isinstance(self.nesterovs_momentum, bool):
            raise ValueError("nesterovs_momentum must be either True or False,"
                             " got %s." % self.nesterovs_momentum)
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False,"
                             " got %s." % self.early_stopping)
        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, "
                             "got %s" % self.validation_fraction)
        if self.beta_1 < 0 or self.beta_1 >= 1:
            raise ValueError("beta_1 must be >= 0 and < 1, got %s" %
                             self.beta_1)
        if self.beta_2 < 0 or self.beta_2 >= 1:
            raise ValueError("beta_2 must be >= 0 and < 1, got %s" %
                             self.beta_2)
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0, got %s." % self.epsilon)
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be > 0, got %s."
                             % self.n_iter_no_change)

        # raise ValueError if not registered
        if self.activation not in ACTIVATIONS:
            raise ValueError("The activation '%s' is not supported. Supported "
                             "activations are %s."
                             % (self.activation, list(sorted(ACTIVATIONS))))
        if self.learning_rate not in ["constant", "invscaling", "adaptive"]:
            raise ValueError("learning rate %s is not supported. " %
                             self.learning_rate)
        supported_solvers = _STOCHASTIC_SOLVERS + ["lbfgs"]
        if self.solver not in supported_solvers:
            raise ValueError("The solver %s is not supported. "
                             " Expected one of: %s" %
                             (self.solver, ", ".join(supported_solvers)))

    def _fit_lbfgs(self, X, y, activations, deltas, coef_grads,
                   intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        opt_res = scipy.optimize.minimize(
                self._loss_grad_lbfgs, packed_coef_inter,
                method="L-BFGS-B", jac=True,
                options={
                    "maxfun": self.max_fun,
                    "maxiter": self.max_iter,
                    "iprint": iprint,
                    "gtol": self.tol
                },
                args=(X, y, activations, deltas, coef_grads, intercept_grads))
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads,
                        intercept_grads, layer_units, incremental):

        if not incremental or not hasattr(self, '_optimizer'):
            params = self.coefs_ + self.intercepts_

            if incremental:
                params = [af.interop.from_ndarray(p)  if isinstance(p, np.ndarray) else p for p in params]

            if self.solver == 'sgd':
                self._optimizer = SGDOptimizer(
                    params, self.learning_rate_init, self.learning_rate,
                    self.momentum, self.nesterovs_momentum, self.power_t)
            elif self.solver == 'adam':
                self._optimizer = AdamOptimizer(
                    params, self.learning_rate_init, self.beta_1, self.beta_2,
                    self.epsilon)

        # early_stopping in partial_fit doesn't make sense
        early_stopping = self.early_stopping and not incremental
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val = train_test_split(
                X, y, random_state=self._random_state,
                test_size=self.validation_fraction,
                stratify=stratify)
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]

        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            for it in range(self.max_iter):
                if self.shuffle:
                    X, y = shuffle(X, y, random_state=self._random_state)
                accumulated_loss = 0.0

                X_af = af.interop.from_ndarray(X)
                y_af = af.interop.from_ndarray(y)

                for batch_slice in gen_batches(n_samples, batch_size):
                    #slow, 33ms
                    #activations[0] = X[batch_slice]
                    activations[0] = X_af[batch_slice]
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X_af[batch_slice], y_af[batch_slice], activations, deltas,
                        coef_grads, intercept_grads)
                    accumulated_loss += batch_loss * (batch_slice.stop -
                                                      batch_slice.start)
                    #TODO: conversion back to np should happen here
                    # update weights

                    #coef_grads = [a.to_ndarray() if a is not None else a for a in coef_grads]
                    #intercept_grads = [np.squeeze(a.to_ndarray()) if a is not None else a for a in intercept_grads]
                    grads = coef_grads + intercept_grads
                    #slow, 16ms
                    self._optimizer.update_params(grads)

                #propagate updates back from optimizer
                #self.coefs_ = [a.to_ndarray() if a is not None else a for a in self.coefs_]
                #self.intercepts_ = [np.squeeze(a.to_ndarray()) if a is not None else a for a in self.intercepts_]
                ncoefs = len(self.coefs_)
                self.coefs_ = self._optimizer.params[:ncoefs]
                self.intercepts_ = self._optimizer.params[ncoefs:]

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_,
                                                         self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = ("Validation score did not improve more than "
                               "tol=%f for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))
                    else:
                        msg = ("Training loss did not improve more than tol=%f"
                               " for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))

                    is_stopping = self._optimizer.trigger_stopping(
                        msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter, ConvergenceWarning)
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        if early_stopping:
            # compute validation score, use that for stopping
            self.validation_scores_.append(self.score(X_val, y_val))

            if self.verbose:
                print("Validation score: %f" % self.validation_scores_[-1])
            # update best parameters
            # use validation_scores_, not loss_curve_
            # let's hope no-one overloads .score with mse
            last_valid_score = self.validation_scores_[-1]

            if last_valid_score < (self.best_validation_score_ +
                                   self.tol):
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy()
                                         for i in self.intercepts_]
        else:
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : returns a trained MLP model.
        """
        ret = self._fit(X, y, incremental=False)
        return ret

    @property
    def partial_fit(self):
        """Update the model with a single iteration over the given data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,)
            The target values.
        Returns
        -------
        self : returns a trained MLP model.
        """
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError("partial_fit is only available for stochastic"
                                 " optimizers. %s is not stochastic."
                                 % self.solver)
        return self._partial_fit

    def _partial_fit(self, X, y):
        return self._fit(X, y, incremental=True)

    def _predict(self, X):
        """Predict using the trained mode
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The decision function of the samples for each class in the model.
        """
        X = check_array(X, accept_sparse=['csr', 'csc'])
        if scipy.sparse.issparse(X):
            X = X.todense()

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + \
            [self.n_outputs_]

        # Initialize layers
        activations = [X]

        for i in range(self.n_layers_ - 1):
            # activations.append(np.empty((X.shape[0],
            # layer_units[i + 1])))
            activations.append(af.constant(0, X.shape[0], layer_units[i + 1]))

        # forward propagate
        activations_af = self._forward_pass_af(activations)
        y_pred = activations_af[-1].to_ndarray()

        return y_pred
