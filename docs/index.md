#
<p align="center">
  <img src="assets/images/banner.svg#only-light" />
</p>
<p align="center">
  <img src="assets/images/banner_dark.svg#only-dark" />
</p>


<br>
<br>

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
## Quick example
```python
import scoringrules as sr
import numpy as np

obs = np.random.randn(100)
fcts = obs[:,None] + np.random.randn(100, 21) * 0.1
sr.crps_ensemble(fcts, obs)
```

## Metrics
- Brier Score
- Continuous Ranked Probability Score (CRPS)
- Logarithmic score
- Error Spread Score
- Energy Score
- Variogram Score

## Acknowledgements
[scoringRules](https://cran.r-project.org/web/packages/scoringRules/index.html) served as a reference for this library. The authors did an outstanding work which greatly facilitated ours. The implementation of the ensemble-based metrics as jit-compiled numpy generalized `ufuncs` was first proposed in [properscoring](https://github.com/properscoring/properscoring), released under Apache License, Version 2.0.
