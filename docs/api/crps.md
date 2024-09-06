# Continuous Ranked Probability Score

Formally, the CRPS is expressed as

$$\text{CRPS}(F, y) = \int_{\mathbb{R}}[F(x)-\mathbb{1}\{y \le x\}]^2 dx$$

where $F(x) = P(X<x)$ is the forecast CDF and $\mathbb{1}\{x \le y\}$ the empirical CDF
of the scalar observation $y$. $\mathbb{1}$ is the indicator function. The CRPS can
also be viewed as the Brier score integrated over all real-valued thresholds.

<!-- When the true forecast CDF is not fully known, but represented by a finite ensemble, the CRPS can be estimated with some error. Several estimators exist and they are implemented in `scoringrules`. For a thorough review of CRPS estimators and their respective biases, refer to Zamo and Naveau (2018)[@zamo_estimation_2018] and Jordan (2016)[@jordan_facets_2016]. -->


## Analytical formulations

::: scoringrules.crps_beta

::: scoringrules.crps_binomial

::: scoringrules.crps_exponential

::: scoringrules.crps_exponentialM

::: scoringrules.crps_gamma

::: scoringrules.crps_gev

::: scoringrules.crps_gpd

::: scoringrules.crps_gtclogistic

::: scoringrules.crps_tlogistic

::: scoringrules.crps_clogistic

::: scoringrules.crps_gtcnormal

::: scoringrules.crps_tnormal

::: scoringrules.crps_cnormal

::: scoringrules.crps_gtct

::: scoringrules.crps_tt

::: scoringrules.crps_ct

::: scoringrules.crps_hypergeometric

::: scoringrules.crps_laplace

::: scoringrules.crps_logistic

::: scoringrules.crps_loglaplace

::: scoringrules.crps_loglogistic

::: scoringrules.crps_lognormal

::: scoringrules.crps_negbinom

::: scoringrules.crps_normal

::: scoringrules.crps_poisson

::: scoringrules.crps_t

::: scoringrules.crps_uniform

::: scoringrules.crps_normal

## Ensemble-based estimators

::: scoringrules.crps_ensemble

::: scoringrules.twcrps_ensemble

::: scoringrules.owcrps_ensemble

::: scoringrules.vrcrps_ensemble

## Quantile-based estimators

::: scoringrules.crps_quantile


<br/><br/>
