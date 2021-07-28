# ArrayFire SKlearn MonkeyPatch

MonkeyPatch sklearn with ArrayFire accelerated variants.

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

To run with time measurements use the flag `-s`

```console
pytest test/test_mlp.py -s
```

--

## TODO

- GradientBoosting
- OneHotEncoder
- RandomForest
- SelectFWE
- QuantileTransformer
- ExtraTreesClassifier
- Imputer
- GenericUnivariateSelect
- SGDClassifier
- LinearSVC
