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

## Patch sklearn

To patch sklearn on import, please install the following package into the venv with installed sklearn and afsklearn packages:

```console
pip install git+https://github.com/roaffix/autowrapt.git@afsklearn
```

--

## TODO

- GradientBoosting
- OneHotEncoder  - Anton
- RandomForest
- SelectFWE
- QuantileTransformer
- ExtraTreesClassifier
- Imputer  - Anton
- GenericUnivariateSelect
- SGDClassifier  - Anton
- LinearSVC  - Stef
- LinearSVR  - Stef
- LogisticRegression  - Stef
- Var threshold  - Anton
