# scoringrules

scoringrules is a python package for the evaluation of probabilistic forecasts. It was inspired by R's [scoringRules](https://cran.r-project.org/web/packages/scoringRules/index.html) packages. It builds on numpy and is accelerated via jit-compilation with numba.


## Features
### Metrics
- Brier Score
- Continuous Ranked Probability Score  (CRPS)
- Energy Score

###
## Installation
```
pip install scoringutils
```

## Documentation

## Acknowledgements
The implementation of the ensemble-based metrics as jit-compiled numpy generalized `ufuncs` was first proposed in [properscoring](https://github.com/properscoring/properscoring), released under Apache License, Version 2.0.
