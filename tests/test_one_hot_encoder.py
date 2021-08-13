import numpy as np
import pytest

from afsklearn.patcher import Patcher
from . import measure_time


def sklearn_example() -> None:
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    X = np.array([['Male', 1], ['Female', 3], ['Female', 2]])
    enc.fit(X)


@pytest.mark.parametrize("n_runs", range(5))
@measure_time
def test_sklearn(n_runs) -> None:
    sklearn_example()


@pytest.mark.parametrize("n_runs", range(5))
@measure_time
def test_afsklearn(n_runs) -> None:
    Patcher.patch("one_hot_encoder")
    sklearn_example()
    Patcher.rollback("one_hot_encoder")


if __name__ == "__main__":
    test_afsklearn()
    test_sklearn()
