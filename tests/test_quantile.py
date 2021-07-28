import numpy as np

from afsklearn.patcher import Patcher
from . import measure_time

rng = np.random.RandomState(0)
X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25000, 100)), axis=0)


def sklearn_example() -> None:
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer()
    qt.fit_transform(X)


@measure_time
def test_sklearn() -> None:
    sklearn_example()


@measure_time
def test_afsklearn() -> None:
    Patcher.patch("quantile_transformer")
    sklearn_example()
    Patcher.rollback("quantile_transformer")


if __name__ == "__main__":
    test_afsklearn()
    # test_sklearn()
