import numpy as np
import pytest

from afsklearn.patcher import Patcher
from . import measure_time


def sklearn_example() -> None:
    from sklearn.impute import SimpleImputer
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(np.array([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]))
    X = np.array([[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]])
    imp_mean.transform(X)


@pytest.mark.parametrize("n_runs", range(5))
@measure_time
def test_sklearn(n_runs) -> None:
    sklearn_example()


@pytest.mark.parametrize("n_runs", range(5))
@measure_time
def test_afsklearn(n_runs) -> None:
    Patcher.patch("simple_imputer")
    sklearn_example()
    Patcher.rollback("simple_imputer")


if __name__ == "__main__":
    test_afsklearn()
    test_sklearn()
