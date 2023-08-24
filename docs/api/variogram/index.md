# Variogram Score 

The varigoram score (VS) is a scoring rule for evaluating multivariate probabilistic forecasts.
It is defined as 

$$\text{VS}_{p}(F, \mathbf{y})= \sum_{i=1}^{d} \sum_{j=1}^{d} \left( \mathbb{E} | X_{i} - X_{j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2}, $$

where $p > 0$, $\mathbf{y} = (y_{1}, \dots, y_{d}) \in \mathbb{R}^{d}$ is the multivariate observation ($d > 1$), and 
$\mathbf{X} = (X_{1}, \dots, X_{d})$ is a random vector that follows the
multivariate forecast distribution $F$ (Scheuerer and Hamill, 2015)[@scheuerer_variogram-based_2015].
The exponent $p$ is typically chosen to be 0.5 or 1.

The variogram score is less sensitive to marginal forecast performance than the energy score,
and Scheuerer and Hamill (2015) argue that it should therefore be more sensitive to errors in the 
forecast's dependence structure.

<br/><br/>

## Ensemble forecasts

While multivariate probabilistic forecasts could belong to a parametric family of 
distributions, such as a multivariate normal distribution, it is more common in practice
that these forecasts are ensemble forecasts; that is, the forecast is comprised of a 
predictive sample $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$, 
where each ensemble member $\mathbf{x}_{i} = (x_{i, 1}, \dots, x_{i, d}) \in \R^{d}$ for
$i = 1, \dots, M$. 

In this case, the expectation in the definition of the variogram score can be replaced by
a sample mean over the ensemble members, yielding the following representation of the variogram
score when evaluating an ensemble forecast $F_{ens}$ with $M$ members:

$$\text{VS}_{p}(F_{ens}, \mathbf{y})= \sum_{i=1}^{d} \sum_{j=1}^{d} \left( \frac{1}{M} \sum_{m=1}^{M} | x_{m,i} - x_{m,j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2}. $$

<br/><br/>

## Weighted variogram scores

It is often the case that certain outcomes are of more interest than others when evaluating 
forecast performance. These outcomes can be emphasised by employing weighted scoring rules.
Weighted scoring rules typically introduce a weight function into conventional scoring rules, 
and users can choose the weight function depending on what outcomes they want to emphasise. 
Allen et al. (2022)[@allen2022evaluating]  introduced three weighted versions of the variogram score. 
These are all available in `scoringrules`. 

Firstly, the outcome-weighted variogram score (see also Holzmann and Klar (2014)[@holzmann2017focusing]) 
is defined as 

$$\text{owVS}_{p}(F, \mathbf{y}; w) = \frac{1}{\bar{w}} \mathbb{E} [ \rho_{p}(\mathbf{X}, \mathbf{y}) w(\mathbf{X}) w(\mathbf{y}) ] - \frac{1}{2 \bar{w}^{2}} \mathbb{E} [ \rho_{p}(\mathbf{X}, \mathbf{X}^{\prime}) w(\mathbf{X}) w(\mathbf{X}^{\prime}) w(\mathbf{y}) ], $$

where

$$ \rho_{p}(\mathbf{x}, \mathbf{z}) = \sum_{i=1}^{d} \sum_{j=1}^{d} \left( |x_{i} - x_{j}|^{p} - |z_{i} - z_{j}|^{p} \right)^{2}, $$

for $\mathbf{x} = (x_{1}, \dots, x_{d}) \in \mathbb{R}^{d}$ and $\mathbf{z} = (z_{1}, \dots, z_{d}) \in \mathbb{R}^{d}$.

Here, $w : \mathbb{R}^{d} \to [0, \infty)$ is the non-negative weight function used to 
target particular multivariate outcomes, and $\bar{w} = \mathbb{E}[w(X)]$.
As before, $\mathbf{X}, \mathbf{X}^{\prime} \sim F$ are independent.

Secondly, Allen et al. (2022) introduced the threshold-weighted variogram score as

$$\text{twVS}_{p}(F, \mathbf{y}; v)= \sum_{i=1}^{d} \sum_{j=1}^{d} \left( \mathbb{E} | v(\mathbf{X})_{i} - v(\mathbf{X})_{j} |^{p} - | v(\mathbf{y})_{i} - v(\mathbf{y})_{j} |^{p} \right)^{2}, $$

where $v : \mathbb{R}^{d} \to \mathbb{R}^{d}$ is a so-called chaining function, so that
$v(\mathbf{X}) = (v(\mathbf{X})_{1}, \dots, v(\mathbf{X})_{d}) \in \mathbb{R}^{d}$. 
The threshold-weighted variogram score transforms the forecasts and observations according 
to the chaining function $v$, prior to calculating the unweighted variogram score. Choosing
a chaining function is generally more difficult than choosing a weight function when
emphasising particular outcomes.

As an alternative, the vertically re-scaled variogram score is defined as 

$$\text{vrVS}_{p}(F, \mathbf{y}; w) = \mathbb{E} [ \rho_{p}(\mathbf{X}, \mathbf{y}) w(\mathbf{X}) w(\mathbf{y}) ] - \frac{1}{2} \mathbb{E} [ \rho_{p}(\mathbf{X}, \mathbf{X}^{\prime}) w(\mathbf{X}) w(\mathbf{X}^{\prime}) ] + \left( \mathbb{E} [ \rho_{p} ( \mathbf{X}, \mathbf{x}_{0} ) w(\mathbf{X}) ] - \rho_{p} ( \mathbf{y}, \mathbf{x}_{0}) w(\mathbf{y}) \right) \left(\mathbb{E}[w(\mathbf{X})] - w(\mathbf{y}) \right), $$

where $w$ and $\rho_{p}$ are as defined above, and $\mathbf{x}_{0} \in \mathbb{R}^{d}$. 
Typically, $\mathbf{x}_{0}$ is chosen to be the zero vector.

Each of these weighted variogram scores targets particular outcomes in a different way. 
Further details regarding the differences between these scoring rules, as well as choices 
for the weight and chaining functions, can be found in Allen et al. (2022). The weighted
variogram scores can easily be computed for ensemble forecasts by
replacing the expectations with sample means over the ensemble members.

<br/><br/>