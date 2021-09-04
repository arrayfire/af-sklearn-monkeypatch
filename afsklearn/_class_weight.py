import numpy as np  # FIXME
from sklearn.utils.validation import _deprecate_positional_args


@_deprecate_positional_args
def compute_class_weight(class_weight, *, classes, y):
    """Estimate class weights for unbalanced datasets.
    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        If 'balanced', class weights will be given by
        ``n_samples / (n_classes * np.bincount(y))``.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.
    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.
    y : array-like of shape (n_samples,)
        Array of original class labels per sample.
    Returns
    -------
    class_weight_vect : ndarray of shape (n_classes,)
        Array with class_weight_vect[i] the weight for i-th class.
    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.
    """
    # Import error caused by circular imports.
    from .preprocessing._label import afLabelEncoder

    if set(y) - set(classes):
        raise ValueError("classes should include all valid labels that can "
                         "be in y")
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        # Find the weight of each class as present in y.
        le = afLabelEncoder()
        y_ind = le.fit_transform(y)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y) / (len(le.classes_) *
                               np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]
    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'balanced', or None,"
                             " got: %r" % class_weight)
        for c in class_weight:
            i = np.searchsorted(classes, c)
            if i >= len(classes) or classes[i] != c:
                raise ValueError("Class label {} not present.".format(c))
            else:
                weight[i] = class_weight[c]

    return weight
