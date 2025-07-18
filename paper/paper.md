---
title: 'scoringrules: A python library for probabilistic forecast evaluation'
tags:
  - python
  - scoring rules
  - probabilistic
  - forecasting
  - prediction
authors:
  - name: Francesco Zanetta
    orcid: 0000-0003-4954-4298
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Sam Allen
    orcid: 0000-0003-1971-8277
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Federal Office of Meteorology and Climatology MeteoSwiss, Zürich, Switzerland
   index: 1
 - name: Institute of Statistics, Karlsruhe Institute of Technology, Karlsruhe, Germany
   index: 2
date: 24 June 2025
bibliography: paper.bib
---

# Summary

Accurate forecasts for future events are essential for effective decision making. It is becoming increasingly common to issue forecasts that are probabilistic, in the form of probability distributions over the possible outcomes. Probabilistic forecasts comprehensively quantify the uncertainty in the prediction, allowing decision makers to balance risks associated with different outcomes. They are therefore routinely reported in a wide variety of application domains, including, but not limited to, economics, oceanography, robotics, criminology, energy, finance, epidemiology, and meteorology.

Probabilistic forecasts can be evaluated using _proper scoring rules_. Informally, a scoring rule is a function that measures how close a forecast distribution is to the corresponding outcome. A scoring rule is called _proper_ if it cannot be improved by hedging the forecast distribution, thus incentivising the forecaster to issue what they truly believe will occur. Proper scoring rules are well established in the forecast evaluation literature as tools for ranking and comparing competing forecasts, and they also provide canonical loss functions when training probabilistic models. Comprehensive reviews of proper scoring rules are provided by @GneitingRaftery2007 and WaghmareZiegel2025, while their use as loss functions is discussed in detail by @DawidEtAl2016.

With the growing list of applications of proper scoring rules, there is an increasing need for software that facilitates their implementation in practice. This paper introduces the `scoringrules` Python package, which provides functionality to calculate several popular proper scoring rules in multiple backends: `NumPy` [@HarrisEtAl2020], `JAX` [@JAX2018], `PyTorch` [@PaszkeEtAl2019], and `TensorFlow` [@AbadiEtAl2016]. The package repository can be found at https://github.com/frazane/scoringrules, with comprehensive documentation available at https://scoringrules.readthedocs.io/en/latest/.



<!-- @PacchiardiEtAl2024 suggests training generative models with proper scoring rules, while @ShenMeinshausen2023 introduce a regression framework based on energy score estimation (see also @ChenEtAl2024). -->



# Statement of need

The original purpose of `scoringrules` was to replicate the functionality of the R package `scoringRules` [@JordanEtAl2019,@Allen2024] in Python, thereby allowing forecasters working in Python to enjoy the same tools as those working in R. While `scoringRules` in R has become a key resource for the forecasting community, no Python package currently exists that offers the same capabilities, despite Python increasingly being adopted for forecasting applications, due partly to its strong support for machine learning. `scoringrules` addresses this.

One particular advantage of `scoringrules` is that, in contrast to other Python packages for forecast evaluation, it supports multiple backends. `NumPy` is the default backend, though performance can be enhanced using Just-In-Time (JIT) compilation in `Numba`,  allowing for fast execution even on large datasets. By additionally offering `JAX`, `PyTorch`, and `TensorFlow` as backends, `scoringrules` can readily be used to define custom loss functions when training machine learning-based probabilistic forecast models.

<!-- The R package `scoringRules` has become a vital tool in the probabilistic forecasting community by facilitating the evaluation of probabilistic forecasts through proper scoring rules, and helping to establish best practices for forecast evaluation. -->

Other Python packages have been developed for forecast evaluation, but they are all different (and generally narrower) in scope. Existing packages focus primarily on the evaluation of point forecasts, generally in the context of weather and climate forecasting, and do not concern the training of probabilistic models. Some related software packages are listed below for comparison:

- `properscoring` [@properscoring] provides two popular scoring rules, both of which are included in `scoringrules`, but is limited in the distributions that it allows as forecasts.
- `xskillscore` [@xskillscore] extends `properscoring` to include several scoring functions for point forecasts, some of which are also included in `scoringrules`, but it does not provide additional functionality to evaluate probabilistic forecasts. `climpred` [@climpred] provides a wrapper for `xskillscore` to evaluate weather and climate forecasts with particular data structures.
- `METplus` [@BrownEtAl2021] provides Python wrappers to implement forecast evaluation metrics from the Model Evaluation Tools (MET) software. However, `METplus` is a comprehensive evaluation system for weather and climate forecasts, whereas `scoringrules` is a more general lightweight library focused on the evaluation of probabilistic forecasts.
- `Pysteps` [@PulkkinenEtAl2019] provides several evaluation metrics for evaluating probabilistic weather fields, but similarly constitutes a comprehensive forecasting system rather than a lightweight evaluation library.
- `PyForecastTools` [@MorleyBurrell2020] provides metrics to evaluate point forecasts, but does not facilitate the evaluation of probabilistic forecasts.
- `epftoolbox` [@LagoEtAl2021] has become a popular Python package for evaluating electricity price forecasts, but similarly does not contain proper scoring rules to evaluate probabilistic forecasts.

Another relevant package is `scores` [@LeeuwenburgEtAl2024], which was originally designed for use within `Jive`, the operational weather forecast evaluation system at the Bureau of Meteorology [@LovedayEtAl2024]. `scores` contains a large collection of evaluation metrics for both point and probabilistic forecasts, but only supports `xarray` datatypes. Several of the scoring rules implemented in the `scores` package listed above call `scoringrules`, thereby providing an `xarray` wrapper. `scoringrules` is therefore indirectly employed within operational weather forecast systems. However, `scoringrules` is applicable in broader generality than weather and climate forecasting, and has additionally been used to evaluate electricity price forecasts [@HirschEtAl2024,Hirsch2025], power forecasts [@Liu2025], and epidemiological forecasts in The Dengue Challenge 2024 [@AraujoEtAl2025].


<!-- Comprehensive documentation is available at https://scoringrules.readthedocs.io/en/latest/. These help pages not only list the scoring rules implemented in the package, but also provide theoretical background on scoring rule properties, including weighted scoring rules and methods for computing scores from finite samples. Additionally, they discuss unbiased estimators for certain proper scoring rules, which are closely related to fair scoring rules as proposed by [@Ferro], and which have recently gained attention as loss functions in AI-based forecasting models [@Lang]. -->


# Available metrics

<!--Functionality is available for popular scoring rules such as the logarithmic score [@Good1952] and the continuous ranked probability score (CRPS) [@MathesonWinkler1976] for univariate outcomes; the energy score, variogram score, and Dawid–Sebastiani score for multivariate outcomes; as well as scoring functions for point forecasts, such as the squared error, quantile (or pinball) score, and interval score, all of which induce proper scoring rules [@Gneiting2011]. -->

<!-- The `scoringrules` package accepts forecasts in the form of parametric distributions or sample-based forecasts—whether univariate or multivariate—alongside arrays of observed outcomes, and returns arrays of scores corresponding to each forecast–observation pair.-->

The `scoringrules` package provides functionality to evaluate probabilistic forecasts for different outcome types. To evaluate probabilistic forecasts for categorical (including binary) outcomes, `scoringrules` offers the Brier score [@Brier1950], logarithmic score, (discrete) ranked probability score [@Epstein1969], and (discrete) ranked logarithmic score [@TodterAhrens2012].

Following @Gneiting2011, point forecasts corresponding to particular functionals of a forecast distribution can be evaluated using consistent scoring functions. The squared error is available for mean forecasts, the absolute error for median forecasts, and the quantile score (or pinball loss) for quantile forecasts. Interval forecasts can additionally be evaluated using the interval (or Winkler) score [@Winkler1972,@Gneiting2007].

Like the `scoringRules` package in R, `scoringrules` provides the logarithmic score [@Good1952] and continuous ranked probability score (CRPS) [@MathesonWinkler1976] for a wide range of parametric forecast distributions when evaluating forecasts for univariate (real-valued) outcomes. A list of the supported distributions can be found in Table 1 of @JordanEtAl2019. The logarithmic score and CRPS can additionally be calculated for forecasts in the form of a predictive sample, obtained, for example, from Markov chain Monte Carlo output, generative models, or from multiple simulations using physical models. Multivariate predictive samples can be evaluated using the energy score [@GneitingRaftery2007], variogram score [@ScheuererHamill2015], and the Gaussian kernel score. Unbiased estimators, or fair versions [@Ferro2014], of these sample-based scoring rules are also available, as well as weighted versions, facilitating flexible user-specified evaluation.

<!--  Multivariate sample forecasts can additionally be evaluated using the energy score, variogram score, Dawid-Sebastiani score, and the Gaussian kernel score. Weighted scoring rules are also available, as has been made available in `scoringRules` [@Allen2024]. -->




# Acknowledgements

The widely-used R package `scoringRules` served as a reference for this library, which greatly facilitated our work. We are additionally grateful for fruitful discussions with the authors, Sebastian Lerch, Fabian Krüger, and Alexander Jordan. The quality of this library has also benefited a lot from discussions with (and contributions from) Nick Loveday and Tennessee Leeuwenburg, whose Python library `scores` similarly provides a comprehensive collection of forecast evaluation methods. Contributions to the package from Simon Hirsch, Daniel Raimundo, and Christopher Bülte are also greatly appreciated. Sam Allen acknowledges funding from the Vector Stiftung through the Young Investigator Group “Artificial Intelligence for Probabilistic Weather Forecasting”.

# References

::: {#refs}
:::
