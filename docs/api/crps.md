# Continuous Ranked Probability Score

Formally, the CRPS is expressed as

$$\text{CRPS}(F, y) = \int_{\mathbb{R}}[F(x)-\mathbb{1}\{y \le x\}]^2 dx$$

where $F(x) = P(X<x)$ is the forecast CDF and $\mathbb{1}\{x \le y\}$ the empirical CDF
of the scalar observation $y$. $\mathbb{1}$ is the indicator function. The CRPS can
also be viewed as the Brier score integrated over all real-valued thresholds.


Citing directly from the abstract of [CITE]:
> The continuous ranked probability score (CRPS) is a much used measure
of performance for probabilistic forecasts of a scalar observation. It is a quadratic
measure of the difference between the forecast cumulative distribution function (CDF)
and the empirical CDF of the observation. Analytic formulations of the CRPS can be
derived for most classical parametric distributions, and be used to assess the efficiency
of different CRPS estimators. When the true forecast CDF is not fully known, but
represented as an ensemble of values, the CRPS is estimated with some error.

Here we implement multiple CRPS estimators based on a finite ensemble in `crps.ensemble()`,
as well as the analytic formulations of the CRPS for some parametric distributions.

<br/><br/>

## Ensemble-based estimators
::: scoringrules.crps.ensemble_int

## Analytical formulations
::: scoringrules.crps.normal
::: scoringrules.crps.lognormal
