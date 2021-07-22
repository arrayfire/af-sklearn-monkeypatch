import arrayfire as af

from sklearn.utils.validation import _deprecate_positional_args
from af_validation import check_consistent_length
from af_validation import check_array
from af_validation import _num_samples

def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    """Check that y_true and y_pred belong to the same regression task
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().
    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'
    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.
    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype: str or list, default="numeric"
        the dtype argument passed to check_array
    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    # irrelevant in af, dim[1] always valid
    #if y_true.numdims() == 1:
        #y_true = y_true.reshape((-1, 1))

    #if y_pred.numdims() == 1:
        #y_pred = y_pred.reshape((-1, 1))

    print(type(y_true))
    print(type(y_pred))
    if y_true.numdims() != 1 and y_pred.numdims() !=1:
        if y_true.shape[1] != y_pred.shape[1]:
            raise ValueError("y_true and y_pred have different number of output "
                             "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = 1 if y_true.numdims() == 1 else  y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput


@_deprecate_positional_args
def r2_score(y_true, y_pred, *, sample_weight=None,
             multioutput="uniform_average"):
    """R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.
    Read more in the :ref:`User Guide <r2_score>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average', \
'variance_weighted'] or None or array-like of shape (n_outputs)
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.
        .. versionchanged:: 0.19
            Default value of multioutput is 'uniform_average'.
    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is
        'raw_values'.
    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).
    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    Examples
    --------
    >>> from sklearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred,
    ...          multioutput='variance_weighted')
    0.938...
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> r2_score(y_true, y_pred)
    1.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [2, 2, 2]
    >>> r2_score(y_true, y_pred)
    0.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [3, 2, 1]
    >>> r2_score(y_true, y_pred)
    -3.0
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = af.sum((weight * (y_true - y_pred) ** 2), dim=0)
    #denominator = (weight * (y_true - np.average(
        #y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                          #dtype=np.float64)
    denominator = af.sum((weight * (y_true - af.tile(af.mean(y_true, weights=sample_weight, dim=0), y_true.shape[0])) ** 2), dim=0)

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    y_sz_1 = 1 if y_true.numdims() == 1 else y_true.shape[1]
    output_scores = af.constant(0, y_sz_1)
    if(af.any_true(valid_score)):
        output_scores[valid_score] = (1.0 - (numerator[valid_score] /
                                            denominator[valid_score])).as_type(output_scores.dtype())
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not af.any_true(nonzero_denominator):
                if not af.any_true(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    #return np.average(output_scores, weights=avg_weights)
    return af.mean(output_scores, weights=avg_weights)


class afRegressorMixin:
    """Mixin class for all regression estimators in scikit-learn."""
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix or a list of generic objects instead,
            shape = (n_samples, n_samples_fitted),
            where n_samples_fitted is the number of
            samples used in the fitting for the estimator.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        Notes
        -----
        The R2 score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {'requires_y': True}


