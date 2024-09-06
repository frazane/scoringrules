<p align="center">
  <img src="docs/assets/images/banner_light.svg#gh-light-mode-only" />
  <img src="docs/assets/images/banner_dark.svg#gh-dark-mode-only" />
</p>

<br>

[![CI](https://github.com/frazane/scoringrules/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/frazane/scoringrules/actions/workflows/ci.yaml)
[![pypi](https://img.shields.io/pypi/v/scoringrules.svg?colorB=<brightgreen>)](https://pypi.python.org/pypi/scoringrules/)

Scoringrules is a python library for evaluating probabilistic forecasts by computing
scoring rules and other diagnostic quantities. It aims to assist forecasting practitioners by
providing a set of tools based the scientific literature and via its didactic approach.

## Features

- **Fast** computations of several probabilistic univariate and multivariate verification metrics
- **Multiple backends**: support for numpy (accelerated with numba), jax, pytorch and tensorflow
- **Didactic approach** to probabilistic forecast evaluation through clear code and documentation

## Installation
Requires python `>=3.10`!

```
pip install scoringrules
```

## Documentation

Learn more about scoringrules in its official documentation at https://frazane.github.io/scoringrules/.


## Quick example
```python
import scoringrules as sr
import numpy as np

obs = np.random.randn(100)
fct = obs[:,None] + np.random.randn(100, 21) * 0.1
sr.crps_ensemble(obs, fct)
```

## Metrics
- Brier Score
- Continuous Ranked Probability Score (CRPS)
- Logarithmic score
- Error Spread Score
- Energy Score
- Variogram Score


## Citation
If you found this library useful for your own research, consider citing:

```
@software{zanetta_scoringrules_2024,
  author = {Francesco Zanetta and Sam Allen},
  title = {Scoringrules: a python library for probabilistic forecast evaluation},
  year = {2024},
  url = {https://github.com/frazane/scoringrules}
}
```

## Acknowledgements
[scoringRules](https://cran.r-project.org/web/packages/scoringRules/index.html) served as a reference for this library. The authors did an outstanding work which greatly facilitated ours. The implementation of the ensemble-based metrics as jit-compiled numpy generalized `ufuncs` was first proposed in [properscoring](https://github.com/properscoring/properscoring), released under Apache License, Version 2.0.
