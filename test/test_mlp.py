import time

from afsklearn.patcher import Patcher
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from . import measure_time

X, y = make_classification(n_samples=10000, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)


def sklearn_example() -> None:
    clf = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, y_train)
    print(f"Score: {clf.score(X_test, y_test)}")


@measure_time
def test_sklearn_mlp() -> None:
    sklearn_example()


@measure_time
def test_afsklearn_mlp() -> None:
    Patcher.patch("mlp_classifier")
    sklearn_example()
    Patcher.rollback("mlp_classifier")


if __name__ == "__main__":
    test_sklearn_mlp()
    test_afsklearn_mlp()
