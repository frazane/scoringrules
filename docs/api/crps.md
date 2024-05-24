# Continuous Ranked Probability Score

Formally, the CRPS is expressed as

$$\text{CRPS}(F, y) = \int_{\mathbb{R}}[F(x)-\mathbb{1}\{y \le x\}]^2 dx$$

where $F(x) = P(X<x)$ is the forecast CDF and $\mathbb{1}\{x \le y\}$ the empirical CDF
of the scalar observation $y$. $\mathbb{1}$ is the indicator function. The CRPS can
also be viewed as the Brier score integrated over all real-valued thresholds.

When the true forecast CDF is not fully known, but represented by a finite ensemble, the CRPS can be estimated with some error. Several estimators exist and they are implemented in `scoringrules`. For a thorough review of CRPS estimators and their respective biases, refer to Zamo and Naveau (2018)[@zamo_estimation_2018] and Jordan (2016)[@jordan_facets_2016].


<h2>Analytical formulations</h2>

::: scoringrules.crps_normal

::: scoringrules.crps_lognormal


<h2>Ensemble-based estimators</h2>

::: scoringrules.crps_ensemble

::: scoringrules.twcrps_ensemble

::: scoringrules.owcrps_ensemble

::: scoringrules.vrcrps_ensemble

<h2>Quantile-based estimators</h2>

::: scoringrules.crps_quantile


<br/><br/>
