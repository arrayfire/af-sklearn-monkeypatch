from setuptools import setup, find_packages

import afsklearn

setup(
    name=afsklearn.__name__,
    version=afsklearn.__version__,
    url='https://github.com/arrayfire/af-sklearn-monkeypatch',
    author='ArrayFire',
    author_email='support@arrayfire.com',
    description='GPU accelerated monkey patch for sklearn',
    packages=find_packages(exclude=['test']),
    install_requires=['arrayfire>=3.8rc1', 'numpy', 'scipy', 'sklearn']
    )
