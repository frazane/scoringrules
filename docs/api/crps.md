# Continuous Ranked Probability Score

The CRPS is defined as

\begin{align}
    \text{CRPS}(F, y) &= \int_{\mathbb{R}}[F(x)-\mathbbm{1}\{y \le x\}]^2 dx \\
    &= \mathbb{E} | X - y | - \frac{1}{2} \mathbb{E} | X - X^{\prime} |
\end{align}

where $F$ is the forecast cumulative distribution function (CDF), $\mathbb{1}$ is the indicator function,
and $X, X^{\prime} \sim F$ are independent. The second expression is only valid when the
forecast distribution $F$ has a finite expectation, $\mathbb{E}|X| < \infty$.

The CRPS can be interpreted as the Brier score between $F(x)$ and $\mathbb{1}\{y \le x\}$,
integrated over all real values $x$. When $F$ is a discrete forecast distribution, such
as an ensemble forecast, the CRPS can be calculated easily be replacing the expectations in
the second expression with sample means over the ensemble members. If $F$ is a point forecast
at some $x \in \mathbb{R}$, then the CRPS reduces the Mean Absolute Error between $x$ and $y$.

<!-- When the true forecast CDF is not fully known, but represented by a finite ensemble, the CRPS can be estimated with some error. Several estimators exist and they are implemented in `scoringrules`. For a thorough review of CRPS estimators and their respective biases, refer to Zamo and Naveau (2018)[@zamo_estimation_2018] and Jordan (2016)[@jordan_facets_2016]. -->


## Analytical formulations

::: scoringrules.crps_beta

::: scoringrules.crps_binomial

::: scoringrules.crps_exponential

::: scoringrules.crps_exponentialM

::: scoringrules.crps_2pexponential

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

::: scoringrules.crps_mixnorm

::: scoringrules.crps_negbinom

::: scoringrules.crps_normal

::: scoringrules.crps_2pnormal

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
