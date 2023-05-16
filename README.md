<h1 align='center'>scoringrules</h1>

Scoringrules is a python package for the evaluation of probabilistic forecasts, inspired by R's [scoringRules](https://cran.r-project.org/web/packages/scoringRules/index.html) package.

<div style="margin-left: 25%;
            margin-right: auto;
            width: 70%">

| :exclamation:  This package is a work in progress, expect breaking changes :exclamation:|
|---------------------------------------------------|

</div>


## Features

- Fast and flexible computations of various metrics
- Didactic approach to probabilistic forecast evaluation and scoring rules
- Easy integration with external libraries (metrics are just numpy universal functions)

## Installation
```
pip install scoringrules
```

## Documentation

Learn more about scoringrules in its official documentation at https://frazane.github.io/scoringrules/.

## Acknowledgements
`scoringRules` served as a reference for this library. The authors did an outstanding work which greatly facilitated ours. The implementation of the ensemble-based metrics as jit-compiled numpy generalized `ufuncs` was first proposed in [properscoring](https://github.com/properscoring/properscoring), released under Apache License, Version 2.0.
