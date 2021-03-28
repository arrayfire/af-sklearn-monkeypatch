from sklearn.neural_network._multilayer_perceptron import MLPClassifier as BaseMLP


class MLPClassifier(BaseMLP):
    def _validate_input(self, X, y, incremental, reset):
        X, y = self._validate_data(
            X, y, accept_sparse=['csr', 'csc'], multi_output=True, dtype=(np.float64, np.float32), reset=reset)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        if (
            (not hasattr(self, "classes_")) or
            (not self.warm_start and not incremental)
        ):
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_
        else:
            classes = unique_labels(y)
            if self.warm_start:
                if set(classes) != set(self.classes_):
                    raise ValueError(
                        f"warm_start can only be used where `y` has the same "
                        f"classes as in the previous call to fit. Previously "
                        f"got {self.classes_}, `y` has {classes}"
                    )
            elif len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                raise ValueError(
                    f"`y` has classes not in `self.classes_`. "
                    f"`self.classes_` has {self.classes_}. 'y' has {classes}."
                )

        # This downcast to bool is to prevent upcasting when working with
        # float32 data
        y = self._label_binarizer.transform(y).astype(bool)
        return X, y

    def predict(self, X):
        """Predict using the multi-layer perceptron classifier
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        check_is_fitted(self)
        y_pred = self._forward_pass_fast(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        return self._label_binarizer.inverse_transform(y_pred)

    @property
    def partial_fit(self):
        """Update the model with a single iteration over the given data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        classes : array of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.
        Returns
        -------
        self : returns a trained MLP model.
        """
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError(
                "partial_fit is only available for stochastic optimizer. %s is not stochastic" % self.solver)
        return self._partial_fit

    def _partial_fit(self, X, y, classes=None):
        if _check_partial_fit_first_call(self, classes):
            self._label_binarizer = LabelBinarizer()
            if type_of_target(y).startswith('multilabel'):
                self._label_binarizer.fit(y)
            else:
                self._label_binarizer.fit(classes)

        super()._partial_fit(X, y)

        return self

    def predict_log_proba(self, X):
        """Return the log of probability estimates.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        log_y_prob : ndarray of shape (n_samples, n_classes)
            The predicted log-probability of the sample for each class
            in the model, where classes are ordered as they are in
            `self.classes_`. Equivalent to log(predict_proba(X))
        """
        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)

    def predict_proba(self, X):
        """Probability estimates.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)
        y_pred = self._forward_pass_fast(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred
