# scoringrules documentation

<br>
<br>

`scoringrules` is a lightweight python library that provides scoring rules to evaluate probabilistic forecasts.

It's original goal was to reproduce the functionality of the R package
[`scoringRules`](https://cran.r-project.org/web/packages/scoringRules/index.html) in python,
thereby allowing forecasting practitioners working in python to enjoy the same tools as those
working in R. The methods implemented in `scoringrules` are therefore based around those
available in `scoringRules`, which are rooted in the scientific literature on probabilistic forecasting.

The scoring rules available in `scoringrules` include, but are not limited to, the

- Brier Score
- Logarithmic Score
- (Discrete) Ranked Probability Score
- Continuous Ranked Probability Score (CRPS)
- Dawid-Sebastiani Score
- Energy Score
- Variogram Score
- Gaussian Kernel Score
- Threshold-Weighted CRPS
- Threshold-Weighted Energy Score
- Quantile Score
- Interval Score




```{toctree}
:hidden:
:caption: Background

theory.md
forecast_dists.md
crps_estimators.md
weighted_scores.md
```

```{toctree}
:hidden:
:caption: Library

user_guide.md
contributing.md
reference.md
```



## Features

- **Fast** computation of several probabilistic univariate and multivariate verification metrics
- **Multiple backends**: support for numpy (accelerated with numba), jax, pytorch and tensorflow
- **Didactic approach** to probabilistic forecast evaluation through clear code and documentation

## Installation
Requires python `>=3.10`!

```
pip install scoringrules
```

## Documentation

Learn more about `scoringrules` in its official documentation at https://scoringrules.readthedocs.io/en/latest/.


## Quick example
```python
import scoringrules as sr
import numpy as np

obs = np.random.randn(100)
fct = obs[:,None] + np.random.randn(100, 21) * 0.1
sr.crps_ensemble(obs, fct)
```

## Citation
If you found this library useful, consider citing:

```
@software{zanetta_scoringrules_2024,
  author = {Francesco Zanetta and Sam Allen},
  title = {scoringrules: a python library for probabilistic forecast evaluation},
  year = {2024},
  url = {https://github.com/frazane/scoringrules}
}
```

## Acknowledgements
- The widely-used R package [`scoringRules`](https://cran.r-project.org/web/packages/scoringRules/index.html)
served as a reference for this library, which greatly facilitated our work. We are additionally
grateful for fruitful discussions with the authors.
- The quality of this library has also benefited a lot from discussions with (and contributions from)
Nick Loveday and Tennessee Leeuwenburg, whose python library [`scores`](https://github.com/nci/scores)
similarly provides a comprehensive collection of forecast evaluation methods.
