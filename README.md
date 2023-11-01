<h1 align='center'>scoringrules</h1>

Scoringrules is a python package for the evaluation of probabilistic forecasts, inspired by R's [scoringRules](https://cran.r-project.org/web/packages/scoringRules/index.html) package.

<div style="margin-left: 25%;
            margin-right: auto;
            width: 70%">

| :exclamation:  This package is a work in progress, expect bugs and breaking changes :exclamation:|
|---------------------------------------------------|

</div>


## Features

- **Fast and flexible** computations of several probabilistic univariate and multivariate verification metrics
- **Multiple backends**: scoringrules supports numpy (accelerated with numba), jax and pytorch, making it useful for optimization
- **Didactic approach** to probabilistic forecast evaluation and scoring rules through clear code and extensive documentation


## Example

```python
import numpy as np
import scoringrules as sr

# dummy data
obs = np.random.randn(10000)
mu = obs + np.random.randn(10000) * 0.1
sigma = abs(np.random.randn(10000)) * 0.3
pred = np.random.randn(10000, 51) * sigma[...,None] + mu[...,None]

# crps estimation for finite ensemble of 51 members
res = sr.crps_ensemble(pred, obs)
print(np.mean(res))
# >>> 0.08141540569209642

# closed form solution
res = sr.crps_normal(mu, sigma, obs)
print(np.mean(res))
# >>> 0.08145091847731219
```
## Installation
```
pip install scoringrules
```

## Documentation

Learn more about scoringrules in its official documentation at https://frazane.github.io/scoringrules/.

## Acknowledgements
[scoringRules](https://cran.r-project.org/web/packages/scoringRules/index.html) served as a reference for this library. The authors did an outstanding work which greatly facilitated ours. The implementation of the ensemble-based metrics as jit-compiled numpy generalized `ufuncs` was first proposed in [properscoring](https://github.com/properscoring/properscoring), released under Apache License, Version 2.0.
