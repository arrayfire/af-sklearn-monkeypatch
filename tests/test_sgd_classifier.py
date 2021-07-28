import numpy as np

from afsklearn.patcher import Patcher

from . import measure_time


def sklearn_example() -> None:
    from sklearn.linear_model import SGDClassifier
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    Y = np.array([1, 1, 2, 2])
    clf = SGDClassifier()
    clf.fit(X, Y)
    print(f"Predict: {clf.predict([[-0.8, -1]])}")


@measure_time
def test_sklearn() -> None:
    sklearn_example()


@measure_time
def test_afsklearn() -> None:
    Patcher.patch("sgd_classifier")
    sklearn_example()
    Patcher.rollback("sgd_classifier")


if __name__ == "__main__":
    test_afsklearn()
    # test_sklearn()
