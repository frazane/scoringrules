# Continuous Ranked Probability Score

Formally, the CRPS is expressed as

$$\text{CRPS}(F, y) = \int_{\mathbb{R}}[F(x)-\mathbb{1}\{y \le x\}]^2 dx$$

where $F(x) = P(X<x)$ is the forecast CDF and $\mathbb{1}\{x \le y\}$ the empirical CDF
of the scalar observation $y$. $\mathbb{1}$ is the indicator function. The CRPS can
also be viewed as the Brier score integrated over all real-valued thresholds.

<br/><br/>

## Ensemble-based estimators

When the true forecast CDF is not fully known, but represented by a finite ensemble, the CRPS can be estimated with some error. Several estimators exist and they are implemented in `scoringrules`. For a thorough review of CRPS estimators and their respective biases, refer to Zamo and Naveau (2018)[@zamo_estimation_2018] and Jordan (2016)[@jordan_facets_2016].

### Integral form (INT)

The numerical approximation of the cumulative integral over the finite ensemble.

$$ \text{CRPS}_{\text{INT}}(M, y) = \int_{\mathbb{R}} \left[ \frac{1}{M}
\sum_{i=1}^M \mathbb{1}\{x_i \le x \} - \mathbb{1}\{y \le x\}  \right] ^2 dx $$

Runs with $O(m\cdot\mathrm{log}m)$ complexity, including the sorting of the ensemble.

### Energy form (NRG)

Introduced by Gneiting and Raftery (2007)[@gneiting_strictly_2007]:

$$ \text{CRPS}_{\text{NRG}}(M, y) = \frac{1}{M} \sum_{i=1}^{M}|x_i - y| - \frac{1}{2 M^2}\sum_{i,j=1}^{M}|x_i - x_j|$$

 It is called the "energy form" because it is the one-dimensional case of the Energy Score.

Runs with $O(m^2)$ complexity.

### Quantile decomposition form (QD)

Introduced by Jordan (2016)[@jordan_facets_2016]:

$$\mathrm{CRPS}_{\mathrm{QD}}(M, y) = \frac{2}{M^2} \sum_{i=1}^{M}(x_i - y)\left[M\mathbb{1}\{y \le x_i\} - i + \frac{1}{2} \right]$$

Runs with $O(m\cdot\mathrm{log}m)$ complexity, including the sorting of the ensemble.

### Probability weighted moment form (PWM)

Introduced by Taillardat et al. (2016)[@taillardat_calibrated_2016]:

$$\mathrm{CRPS}_{\mathrm{NRG}}(M, y) = \frac{1}{M} \sum_{i=1}^{M}|x_i - y| + \hat{\beta_0} - 2\hat{\beta_1},$$

where $\hat{\beta_0} = \frac{1}{M} \sum_{i=1}^{M}x_i$ and $\hat{\beta_1} = \frac{1}{M(M-1)} \sum_{i=1}^{M}(i - 1)x_i$. Runs with $O(m\cdot\mathrm{log}m)$ complexity, including the sorting of the ensemble.


<br/><br/>

::: scoringrules.crps.ensemble

## Analytical formulations
::: scoringrules.crps.normal
::: scoringrules.crps.lognormal
