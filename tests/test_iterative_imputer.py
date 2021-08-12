import numpy as np

from afsklearn.patcher import Patcher
from . import measure_time


def sklearn_example() -> None:
    from sklearn.impute._iterative import IterativeImputer
    imp_mean = IterativeImputer(random_state=0)
    imp_mean.fit(np.array([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]))
    X = np.array([[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]])
    imp_mean.transform(X)


@measure_time
def test_sklearn() -> None:
    sklearn_example()


@measure_time
def test_afsklearn() -> None:
    Patcher.patch("iterative_imputer")
    sklearn_example()
    Patcher.rollback("iterative_imputer")


if __name__ == "__main__":
    test_afsklearn()
    test_sklearn()
