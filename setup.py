from setuptools import setup, find_packages

import af-sklearn-monkeypatch as afskmp

setup(
    name=afskmp.__name__,
    version=afskmp.__version__,
    url='https://github.com/arrayfire/af-sklearn-monkeypatch',
    author='ArrayFire',
    author_email='support@arrayfire.com',
    description='GPU accelerated monkey patch for sklearn',
    packages=find_packages(exclude=['test']),
    install_requires=['arrayfire>=3.8', 'numpy', 'scipy', 'sklearn']
    )
