# ArrayFire SKlearn MonkeyPatch

MonkeyPatch sklearn with ArrayFire accelerated variants. Tested classifiers match sklearn interface and pass sklearn tests. Currently targeting scikit-learn 0.22.
Patching sklearn components can be done explicitly with the Patcher class as follows:
```
import sklearn
from sklearn.neural_network import MLPClassifier

from afsklearn.patcher import Patcher
Patcher.patch("mlp_classifier") # patches MLPClassifier with accelerated variant

clf = MLPClassifier(random_state=1, max_iter=300) # accelerated arrayfire classifier

Patcher.rollback("mlp_classifier")
# returns sklearn package to default state w/o arrayfire
```
Instead of manually replacing individual classifiers, all possible components can be replaced at once:
```
import sklearn
from afsklearn.patcher import Patcher

Patcher.patch_all() # patches scikit-learn with all accelerated classifiers
# sklearn functions here
Patcher.rollback_all() #returns sklearn package to default state w/o arrayfire
```
In the case that no code modification is desired, the [autowrapt](https://github.com/syurkevi/autowrapt) package can be used to globally and automatically replace sklearn during python's import. After installing the linked autowrapt package, set the `AUTOWRAPT_BOOTSTRAP=afsklearn` environment variable to enable the import hooks.

## Installation

```console
pip install -r requirements.txt
```

## Tests

To run all tests

```console
pytest .
```

To run specific test

```console
pytest tests/test_mlp.py
```
