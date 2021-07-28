from afsklearn.patcher import Patcher

from . import measure_time


def sklearn_example() -> None:
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)


@measure_time
def test_sklearn() -> None:
    sklearn_example()


@measure_time
def test_afsklearn() -> None:
    Patcher.patch("one_hot_encoder")
    sklearn_example()
    Patcher.rollback("one_hot_encoder")


if __name__ == "__main__":
    test_afsklearn()
    test_sklearn()
