import numpy as np  # FIXME
from sklearn.utils.validation import _num_samples

from .._encode import _encode, _unique
from .._validation import check_is_fitted, column_or_1d
from ..base import afBaseEstimator, afTransformerMixin


class afLabelEncoder(afTransformerMixin, afBaseEstimator):
    """Encode target labels with value between 0 and n_classes-1.
    This transformer should be used to encode target values, *i.e.* `y`, and
    not the input `X`.
    Read more in the :ref:`User Guide <preprocessing_targets>`.
    .. versionadded:: 0.12
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.
    Examples
    --------
    `LabelEncoder` can be used to normalize labels.
    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])
    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
    See Also
    --------
    OrdinalEncoder : Encode categorical features using an ordinal encoding
        scheme.
    OneHotEncoder : Encode categorical features as a one-hot numeric array.
    """

    def fit(self, y):
        """Fit label encoder.
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = _unique(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = _unique(y, return_inverse=True)
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        return _encode(y, uniques=self.classes_)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.
        Returns
        -------
        y : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff):
            raise ValueError(
                "y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]

    def _more_tags(self):
        return {'X_types': ['1dlabels']}
