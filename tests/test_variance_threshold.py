from afsklearn.patcher import Patcher

from timing_utils import measure_time
import numpy as np

X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
X = np.tile(X, (10000, 100))
print(X.shape)


def sklearn_example() -> None:
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    selector.fit_transform(X)


@measure_time
def test_sklearn() -> None:
    sklearn_example()


@measure_time
def test_afsklearn() -> None:
    Patcher.patch("variance_threshold")
    sklearn_example()
    Patcher.rollback("variance_threshold")


if __name__ == "__main__":
    test_afsklearn()
    test_sklearn()
