# Energy Score

The energy score (ES) is a scoring rule for evaluating multivariate probabilistic forecasts.
It is defined as

$$\text{ES}(F, \mathbf{y})= \mathbb{E} \| \mathbf{X} - \mathbf{y} \| - \frac{1}{2} \mathbb{E} \| \mathbf{X} - \mathbf{X}^{\prime} \|, $$

where $\mathbf{y} \in \mathbb{R}^{d}$ is the multivariate observation ($d > 1$), and
$\mathbf{X}$ and $\mathbf{X}^{\prime}$ are independent random variables that follow the
multivariate forecast distribution $F$ (Gneiting and Raftery, 2007)[@gneiting_strictly_2007].
If the dimension $d$ were equal to one, the energy score would reduce to the continuous ranked probability score (CRPS).

While multivariate probabilistic forecasts could belong to a parametric family of
distributions, such as a multivariate normal distribution, it is more common in practice
that these forecasts are ensemble forecasts; that is, the forecast is comprised of a
predictive sample $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$,
where each ensemble member $\mathbf{x}_{1}, \dots, \mathbf{x}_{M} \in \R^{d}$.

In this case, the expectations in the definition of the energy score can be replaced by
sample means over the ensemble members, yielding the following representation of the energy
score when evaluating an ensemble forecast $F_{ens}$ with $M$ members.


::: scoringrules.energy_score

<h2>Weighted versions</h2>

The energy score provides a measure of overall forecast performance. However, it is often
the case that certain outcomes are of more interest than others, making it desirable to
assign more weight to these outcomes when evaluating forecast performance. This can be
achieved using weighted scoring rules. Weighted scoring rules typically introduce a
weight function into conventional scoring rules, and users can choose the weight function
depending on what outcomes they want to emphasise. Allen et al. (2022)[@allen2022evaluating]
discuss three weighted versions of the energy score. These are all available in `scoringrules`.

Firstly, the outcome-weighted energy score (originally introduced by Holzmann and Klar (2014)[@holzmann2017focusing])
is defined as

$$\text{owES}(F, \mathbf{y}; w)= \frac{1}{\bar{w}} \mathbb{E} \| \mathbf{X} - \mathbf{y} \| w(\mathbf{X}) w(\mathbf{y}) - \frac{1}{2 \bar{w}^{2}} \mathbb{E} \| \mathbf{X} - \mathbf{X}^{\prime} \| w(\mathbf{X})w(\mathbf{X}^{\prime})w(\mathbf{y}), $$

where $w : \mathbb{R}^{d} \to [0, \infty)$ is the non-negative weight function used to
target particular multivariate outcomes, and $\bar{w} = \mathbb{E}[w(X)]$.
As before, $\mathbf{X}, \mathbf{X}^{\prime} \sim F$ are independent.

::: scoringrules.owenergy_score

<br/><br/>

Secondly, Allen et al. (2022) introduced the threshold-weighted energy score as

$$\text{twES}(F, \mathbf{y}; v)= \mathbb{E} \| v(\mathbf{X}) - v(\mathbf{y}) \| - \frac{1}{2} \mathbb{E} \| v(\mathbf{X}) - v(\mathbf{X}^{\prime}) \|, $$

where $v : \mathbb{R}^{d} \to \mathbb{R}^{d}$ is a so-called chaining function.
The threshold-weighted energy score transforms the forecasts and observations according
to the chaining function $v$, prior to calculating the unweighted energy score. Choosing
a chaining function is generally more difficult than choosing a weight function when
emphasising particular outcomes.

::: scoringrules.twenergy_score

<br/><br/>

As an alternative, the vertically re-scaled energy score is defined as

$$
\begin{split}
    \text{vrES}(F, \mathbf{y}; w, \mathbf{x}_{0}) = & \mathbb{E} \| \mathbf{X} - \mathbf{y} \| w(\mathbf{X}) w(\mathbf{y}) \\ & - \frac{1}{2} \mathbb{E} \| \mathbf{X} - \mathbf{X}^{\prime} \| w(\mathbf{X})w(\mathbf{X}^{\prime}) \\
    & + \left( \mathbb{E} \| \mathbf{X} - \mathbf{x}_{0} \| w(\mathbf{X}) - \| \mathbf{y} - \mathbf{x}_{0} \| w(\mathbf{y}) \right) \left(\mathbb{E}[w(\mathbf{X})] - w(\mathbf{y}) \right),
\end{split}
$$

where $w : \mathbb{R}^{d} \to [0, \infty)$ is the non-negative weight function used to
target particular multivariate outcomes, and $\mathbf{x}_{0} \in \mathbb{R}^{d}$. Typically,
$\mathbf{x}_{0}$ is chosen to be zero.


::: scoringrules.vrenergy_score

<br/><br/>

Each of these weighted energy scores targets particular outcomes in a different way.
Further details regarding the differences between these scoring rules, as well as choices
for the weight and chaining functions, can be found in Allen et al. (2022). The weighted
energy scores can easily be computed for ensemble forecasts by
replacing the expectations with sample means over the ensemble members.

<br/><br/>
