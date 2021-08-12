import numpy as np
import pytest

from afsklearn.patcher import Patcher
from . import measure_time

X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])


def sklearn_example() -> None:
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    selector.fit_transform(X)


@pytest.mark.parametrize("n_runs", range(5))
@measure_time
def test_sklearn(n_runs) -> None:
    sklearn_example()


@pytest.mark.parametrize("n_runs", range(5))
@measure_time
def test_afsklearn(n_runs) -> None:
    Patcher.patch("variance_threshold")
    sklearn_example()
    Patcher.rollback("variance_threshold")


if __name__ == "__main__":
    test_afsklearn()
    test_sklearn()
