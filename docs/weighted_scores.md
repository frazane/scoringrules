(weighted_scores)=
# Weighted scoring rules

Most scoring rules used in practice evaluate the entire forecast distribution. However,
it is often the case that some outcomes are of more interest than others. For example, extreme
outcomes often have a large impact on forecast users, making forecasts for these outcomes
particularly valuable. One could therefore argue that these outcomes should be assigned a
higher weight when evaluating forecast performance. For example, Savage (1971) remarks
that "the scoring rule should encourage the respondent to work hardest at what
the questioner most wants to know." For this purpose, weighted scoring rules
have been introduced, which extend conventional scoring rules to incorporate a weight function
into the score. This weight function can then be chosen to emphasise the outcomes that
are of most interest to the forecast users, facilitating flexible, user-oriented evaluation.

## The forecaster's dilemma

Given a proper scoring rule $S$ on $\Omega$, one may try to emphasise particular outcomes by multiplying the
score by a weight depending on what outcome occurs. That is, to evaluate the forecast $F$ using

$$ w(y)S(F, y), $$

for some non-negative weight function $w : \Omega \to [0, \infty)$. In this case, the score will
be scaled depending on what outcome occurs.

However, Gneiting and Ranjan (2011) demonstrate that this is not a proper scoring rule. In particular,
if $Y \sim G$, then the expected score $\mathbb{E}[w(Y)S(F, Y)]$ is minimised by a weighted version of
$G$ rather than $G$ itself. This weighted version of $G$ assigns higher probability density to the
outcomes that are assigned higher weight by $w$. For example, if $G$ emits a density $g$, then
the weighted score is minimised by the distribution $G_{w}$ with density

$$ g_{w}(x) = \frac{w(x)g(x)}{\int_{\Omega} w(z) g(z) dz}. $$

If the weight function is of the form $w(x) = \mathbb{1}\{x \in A\}$ for some subset of the outcomes
$A \subset \Omega$, then $G_{w}$ is the conditional distribution of $Y$ given that $Y \in A$.
Note that this easily extends to distributions that do not emit a density function.

For more intuition, consider the case where the weight function restricts attention to extreme outcomes.
If we only evaluate forecasts made for extreme outcomes, then it becomes an attractive strategy to
always predict that an extreme event will occur, regardless of how likely such an event is.
However, such a forecast is useless in practice: if we always predict that an extreme
event will occur, then we cannot use this forecast to distinguish whether or not an extreme event will actually occur.
Lerch et al. (2017) term this the *forecaster's dilemma*. Instead, alternative, more
theoretically-desirable methods are required to emphasise particular outcomes when evaluating
forecast performance.


## Outcome-weighting (conditional scores)

Holzmann and Klar (2017) remarked that if the expected scoring rule is minimised by a weighted
version of the true distribution of $Y$, then we can circumvent the forecaster's dilemma by
evaluating forecasts via their weighted version. That is, given a proper scoring rule $S$,
the weighted scoring rule

$$ \mathrm{ow}S(F, y) = w(y)S(F_{w}, y) $$

is minimised when $F_{w} = G_{w}$, which is obviously the case (though generally not uniquely) when $F = G$.
This scoring rule is therefore generally proper but not strictly proper.

This approach was initially applied to the Log score by Diks et al. (2011), but can be implemented
using any proper scoring rule, providing a very general framework
with which to weight particular outcomes during forecast evaluation whilst retaining the
propriety of the scoring rule. We refer to these scoring rules as outcome-weighted
scoring rules, since they weight the score depending on the outcome. They have also been
referred to as conditional scores, since, when the weight function is of the form
$w(x) = \mathbb{1}\{x \in A\}$, they evaluate the forecast via its conditional distribution
on $A$.

In evaluating the conditional forecast distribution, these outcome-weighted scoring rules do not
take into account the forecast probability that an outcome of interest will occur. For example,
if $w(x) = \mathbb{1}\{x \in A\}$, then the score will not depend on the forecast probability
that $\mathbb{1}\{Y \notin A\}$. Two forecasts that have the same conditional distribution will
therefore receive the same score.

Holzmann and Klar (2017) proposed to remedy this by adding a proper scoring rule for binary outcomes,
such as the Brier score, to the outcome-weighted scoring rule. However, there is no guarantee that
the scale of the two scores are compatible, and there is often no canonical choice for the scoring
rule for binary outcomes; the Log score is an exception, since this approach can recover the censored
likelihood score introduced by Diks et al. (2011), which is discussed later (Holzmann and Klar, 2017, Example 4).
Hence, these complemented outcome-weighted scores are not available in `scoringrules`. However,
`scoringrules` does offer outcome-weighted versions of some popular scoring rules, which
are listed in the following sections.


## Threshold-weighting (censored scores)

When $\Omega \subseteq \mathbb{R}$, Matheson and Winkler (1976) propose evaluating probabilistic
forecasts by considering the probability forecast that a threshold will be exceeded, evaluating
this forecast using a proper scoring rules for binary outcomes, and then integrating over all thresholds.
For example, they introduce the Continuous Ranked Probability Score (CRPS) as

$$
    \mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbb{1}\{y \le z\})^{2} dz,
$$

which is the integral of the Brier score. Matheson and Winkler (1976) note that the integral
over the binary scoring rule could be defined more generally with respect to any probability
distribution, essentially incorporating a non-negative weight $w(z)$ into the integrand.
They then remark that "if certain regions of values of the variable are of particular interest,
the experimenter might make [$w(z)$] higher in these regions than it is elsewhere."

In the case of the CRPS, Gneiting and Ranjan (2011) refer to this weighted version as the
*threshold-weighted CRPS*, since the weight function emphasises particular thresholds at
which the binary scoring rules are evaluated. Todter and Ahrens (2012) replaced the Brier
score with the Log score and similarly introduced a *Continuous Ranked Logarithmic Score (CRLS)*,
as well as a *threshold-weighted CRLS*.

Allen et al. (2024) demonstrate that the threshold-weighted CRPS corresponds to transforming
the forecast and observation, before evaluating the transformed forecast against the transformed
observation using the standard (unweighted) CRPS; the transformation is determined by the weight
function. For example, when $w(x) = \mathbb{1}\{x > t\}$, for $t \in \mathbb{R}$, one possible
transformation is $v(x) = \max\{x, t\}$. In this case, the forecast distribution and the observation
are essentially censored at the threshold $t$, which is akin to the censored likelihood score
proposed by Diks et al. (2011).

Moreover, while the original idea of threshold-weighting was implicitly univariate and specific
to scoring rules constructed using the approach of Matheson and Winkler (1976) described above,
the idea of transforming forecasts and observations prior to evaluation can easily be implemented
on arbitrary outcome spaces $\Omega$, and with arbitrary scoring rules. This facilitates the
introduction of threshold-weighted versions of any proper scoring rule. Some examples constructed
using popular scoring rules are listed in the following sections.


## Vertical re-scaling

Outcome-weighted and threshold-weighted scoring rules involve transforming the forecast distributions
and outcomes (e.g. via conditioning or censoring) before calculating the original unweighted scores.
Alternatively, since many scoring rules are defined in terms of (dis-)similarity metrics, we could
instead define weighted scoring rules by transforming the output of these metrics, rather than the
inputs. In particular, *kernel scores* are scoring rules of the form

$$
    S_{k}(F, y) = \frac{1}{2} k(y, y) + \frac{1}{2} \mathbb{E} k(X, X^{\prime}) - \mathbb{E} k(X, y),
$$

where $k$ is a positive definite kernel on $\Omega$. A positive definite kernel can be interpreted
as a generalised inner product, so that $k(x, x^{\prime})$, $x, x^{\prime} \in \Omega$, loosely
represents a measure of similarity between $x$ and $x^{\prime}$.

Rasmussen and Williams (2006) remark that if $k$ is a positive definite kernel on $\Omega$, then
so is the weighted kernel $\check{k}(x, x^{\prime}) = k(x, x^{\prime})w(x)w(x^{\prime})$, for
some non-negative weight function $w$ on $\Omega$. This can therefore be implemented within
the definition of kernel scores above to obtain weighted versions of any kernel score.
Allen et al. (2024) term these *vertically re-scaled kernel scores*, since they involve
re-scaling the output of the similarity measure.

For an interpretation of these vertically re-scaled kernel scores, consider how $\check{k}$ adapts $k$.
The similarity between $x$ and $x^{\prime}$ (as measured using $k$) is weighted depending on their values.
For example, if $w(x) = \mathbb{1}\{x \in A\}$ for some subset of the outcomes $A \subset \Omega$, then

$$
\check{k}(x, x^{\prime}) =
\begin{cases}
    k(x, x^{\prime}), & x \in A, x^{\prime} \in A, \\
    0, & \text{otherwise.}
\end{cases}
$$

In this case, the measure of similarity is restricted to instances where both inputs are in the region
$A$ of interest. In the context of kernel scores, consider the term $\mathbb{E} \check{k}(X, y)$.
When the outcome $y \notin A$, this term is equal to zero, regardless of the forecast.
When $y \in A$, the term is maximised (resulting in a lower score) when $F$ assigns a higher probability
to the region $A$. Conversely, the second term $\mathbb{E} \check{k}(X, X^{\prime})$ does not depend on the
observation, and is minimised (resulting in a lower score) when $F$ assigns lower probability to $A$.
Hence, this vertically re-scaled scoring rule rewards forecasts that issue a high probability to $A$
when $y \in A$, and a low probability to $A$ otherwise.

Similarly to the threshold-weighted scores, the behaviour of the forecast outside of $A$ does not
contribute to the score beyond the probability that is assigned to $A$. In many relevant cases,
vertical re-scaling is equivalent to threshold-weighting (Allen et al., 2024, Proposition 4.10).


## Examples

In the following, let $w : \Omega \to [0, \infty)$ be a weight function that
assigns a non-negative weight to each possible outcome, unless specified otherwise.

### Log score

Assuming the forecast distribution $F$ emits a density function $f$ on $\Omega$, the Log score is defined as

$$ \mathrm{LS}(F, y) = -\log f(y). $$

The *conditional likelihood score* is defined as

$$
\begin{align}
\mathrm{coLS}(F, y; w) &= - w(y) \log \left( \frac{f(y)}{\int_{\Omega} w(z)f(z) dz} \right) \\
    &= - w(y) \log f(y) + w(y) \log \left[ \int_{\Omega} w(z)f(z) dz \right].
\end{align}
$$

This can be interpreted as an outcome-weighted version of the Log score (Holzmann and Klar, 2017, Example 1).

For a weight function $w : \Omega \to [0, 1]$, the *censored likelihood score* is defined as

$$
\mathrm{ceLS}(F, y; w) = - w(y) \log f(y) - (1 - w(y)) \log \left[ 1 - \int_{\Omega} w(z)f(z) dz \right].
$$

The censored likelihood score evaluates the forecast via the censored density function, and can
therefore be interpreted as a threshold-weighted version of the Log score (see the discussion
in de Punder et al., 2023).

Since the Log score is not a kernel score, no vertically re-scaled version of the score exists.



### Kernel scores

The kernel score corresponding to a positive definite kernel $k$ on $\Omega$ is defined as

$$
    S_{k}(F, y) = \frac{1}{2} k(y, y) + \frac{1}{2} \mathbb{E} k(X, X^{\prime}) - \mathbb{E} k(X, y),
$$

where $X, X^{\prime} \sim F$ are independent. Allen et al. (2024) discuss how weighted versions
of kernel scores can be constructed. Since many popular scoring rules are kernel scores,
including the CRPS, energy score, and Variogram score,
this facilitates the introduction of weighted versions of these popular scores. These
examples are given in the following sections.

The outcome-weighted version of a kernel score can be written as

$$
    \mathrm{ow}S_{k}(F, y; w) = \frac{1}{2} k(y, y)w(y) + \frac{1}{2 \bar{w}_{F}^{2}} \mathbb{E} k(X, X^{\prime})w(X)w(X^{\prime})w(y) - \frac{1}{\bar{w}_{F}}\mathbb{E} k(X, y)w(X)w(y),
$$

where $\bar{w}_{F} = \mathbb{E}[w(X)]$.

The threshold-weighted version of a kernel score can be written as

$$
    \mathrm{tw}S_{k}(F, y; v) = \frac{1}{2} k(v(y), v(y)) + \frac{1}{2} \mathbb{E} k(v(X), v(X^{\prime})) - \mathbb{E} k(v(X), v(y)),
$$

where $v : \Omega \to \Omega$ is termed the *chaining function*. When $\Omega \subseteq \mathbb{R}$,
the chaining function can be defined as the anti-derivative of the weight function. That is,
$v(x) - v(x^{\prime}) = \int_{x^{\prime}}^{x} w(z) dz$ for all $x, x^{\prime} \in \mathbb{R}$.
However, more generally, there is no canonical way to map a weight function $w$ to a chaining function $v$.

The vertically re-scaled version of a kernel score can be written as

$$
    \mathrm{vr}S_{k}(F, y; w) = \frac{1}{2} k(y, y)w(y)^{2} + \frac{1}{2} \mathbb{E} k(X, X^{\prime})w(X)w(X^{\prime}) - \mathbb{E} k(X, y)w(X)w(y).
$$

For any positive definite kernel $k$ on $\Omega$, the functions $\tilde{k}(x, x^{\prime}) = k(v(x), v(x^{\prime}))$
and $\check{k}(x, x^{\prime}) = k(x, x^{\prime})w(x)w(x^{\prime})$ are also positive definite
kernels on $\Omega$ (Rasmussen and Williams, 2006). Hence, the threshold-weighted and vertically
re-scaled versions of kernel scores are themselves kernel scores.


### CRPS

The CRPS is defined as

$$
\begin{align}
\mathrm{CRPS}(F, y) &= \int_{-\infty}^{\infty} (F(z) - \mathbb{1}\{y \le z\})^{2} dz \\
&= \mathbb{E}|X - y| - \frac{1}{2} \mathbb{E}|X - X^{\prime}|,
\end{align}
$$
where $y \in \Omega \subseteq \mathbb{R}$, and $X, X^{\prime} \sim F$ are independent
(Matheson and Winkler, 1976; Gneiting and Rafery, 2007).

Holzmann and Klar (2017) introduce the *outcome-weighted CRPS* as

$$
\begin{align}
\mathrm{owCRPS}(F, y; w) &= w(y) \int_{-\infty}^{\infty} (F_{w}(z) - \mathbb{1}\{y \le z\})^{2} dz \\
&= \frac{1}{\bar{w}_{F}}\mathbb{E}|X - y|w(X)w(y) - \frac{1}{2\bar{w}_{F}^{2}} \mathbb{E}|X - X^{\prime}|w(X)w(X^{\prime})w(y),
\end{align}
$$
where $\bar{w}_{F} = \mathbb{E}[w(X)]$.

Gneiting and Ranjan (2011) introduce the *threshold-weighted CRPS* as

$$
\begin{align}
\mathrm{twCRPS}(F, y; w) &= \int_{-\infty}^{\infty} (F(x) - 1\{y \le x\})^{2} w(x) dx \\
&= \mathbb{E} | v(X) - v(y) | - \frac{1}{2} \mathbb{E} |v(X) - v(X^{\prime}) |,
\end{align}
$$

where $v : \mathbb{R} \to \mathbb{R}$ is an anti-derivative of the weight function $w$ (Allen et al., 2024).

Allen et al. (2023) introduce the *vertically re-scaled CRPS* as

$$
\mathrm{vrCRPS}(F, y; w, x_{0}) = \mathbb{E} | X - y |w(X)w(y) - \frac{1}{2} \mathbb{E} | X - X^{\prime} | w(X)w(X^{\prime}) + \left( \mathbb{E}|X - x_{0}|w(X) - |y - x_{0}|w(y) \right) \left( \mathbb{E}w(X) - w(y) \right),
$$

for some $x_{0} \in \mathbb{R}$. The canonical choice is $x_{0} = 0$, though if the weight function
is of the form $w(x) = \mathbb{1}\{x > t\}$ or $w(x) = \mathbb{1}\{x < t\}$, for some threshold
$t \in \mathbb{R}$, then setting $x_{0} = t$ recovers the threshold-weighted CRPS
(Allen et al., 2024, Proposition 4.10).


### Energy score

The energy score is defined as

$$
\mathrm{ES}(F, \boldsymbol{y}) = \mathbb{E} \| \boldsymbol{X} - \boldsymbol{y} \| - \frac{1}{2} \mathbb{E} \| \boldsymbol{X} - \boldsymbol{X}^{\prime} \|,
$$

where $\| \cdot \|$ is the Euclidean distance on $\mathbb{R}^{d}$,
$\boldsymbol{y} \in \Omega \subseteq \mathbb{R}^{d}$, and $\boldsymbol{X}, \boldsymbol{X}^{\prime} \sim F$ are independent,
with $F$ a multivariate predictive distribution on $\Omega$ (Gneiting and Raftery, 2007).

The *outcome-weighted energy score* is similarly defined as

$$
\mathrm{owES}(F, \boldsymbol{y}; w) = \frac{1}{\bar{w}_{F}} \mathbb{E} \| \boldsymbol{X} - \boldsymbol{y} \| w(\boldsymbol{X})w(\boldsymbol{y}) - \frac{1}{2\bar{w}_{F}^{2}} \mathbb{E} \| \boldsymbol{X} - \boldsymbol{X}^{\prime} \|w(\boldsymbol{X})w(\boldsymbol{X}^{\prime})w(\boldsymbol{y}),
$$

where $\boldsymbol{X}, \boldsymbol{X}^{\prime} \sim F$ are independent, and
$\bar{w}_{F} = \mathbb{E}[w(\boldsymbol{X})]$ for some weight function $w : \Omega \to [0, \infty)$
(Holzmann and Klar, 2017).

The *threshold-weighted energy score* constitutes a multivariate extension of the threshold-weighted CRPS,

$$
\mathrm{twES}(F, y\boldsymbol{y}; v) = \mathbb{E} | v(\boldsymbol{X}) - v(\boldsymbol{y}) | - \frac{1}{2} \mathbb{E} |v(\boldsymbol{X}) - v(\boldsymbol{X}^{\prime}) |,
$$

where $v : \mathbb{R} \to \mathbb{R}$,

and the *vertically re-scaled energy score* constitutes a multivariate extension of the vertically re-scaled CRPS,

$$
\mathrm{vrES}(F, \boldsymbol{y}; w, \boldsymbol{x}_{0}) = \mathbb{E} | \boldsymbol{X} - \boldsymbol{y} |w(X)w(\boldsymbol{y}) - \frac{1}{2} \mathbb{E} | \boldsymbol{X} - \boldsymbol{X}^{\prime} | w(\boldsymbol{X})w(\boldsymbol{X}^{\prime}) + \left( \mathbb{E}|\boldsymbol{X} - \boldsymbol{x}_{0}|w(\boldsymbol{X}) - |\boldsymbol{y} - \boldsymbol{x}_{0}|w(\boldsymbol{y}) \right) \left( \mathbb{E}w(\boldsymbol{X}) - w(\boldsymbol{y}) \right)
$$

(Allen et al., 2024). As with the vertically re-scaled CRPS, the canonical choice is $\boldsymbol{x}_{0} = \boldsymbol{0}$,
though if the weight function is of the form $w(\boldsymbol{x}) = \mathbb{1}\{\boldsymbol{x} > \boldsymbol{x}\}$ or
$w(\boldsymbol{x}) = \mathbb{1}\{\boldsymbol{x} < t\}$, for some multivariate threshold
$\boldsymbol{x} \in \mathbb{R}$ (with $<$ and $>$ understood componentwise), then setting
$\boldsymbol{x}_{0} = \boldsymbol{t}$ recovers the threshold-weighted energy score (Allen et al., 2024, Proposition 4.10).


### Variogram score

The Variogram score is defined as

$$
    \mathrm{VS}_{p}(F, \boldsymbol{y}) = \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left( \mathbb{E} | X_{i} - X_{j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2},
$$

where $p > 0$, $\boldsymbol{y} = (y_{1}, \dots, y_{d}) \in \Omega \subseteq \mathbb{R}^{d}$,
$\boldsymbol{X} = (X_{1}, \dots, X_{d}) \sim F$, with $F$ a multivariate predictive distribution on $\Omega$,
and $h_{i,j} \ge 0$ are weights assigned to different pairs of dimensions (Scheuerer and Hamill, 2015).

Allen (2024) writes the *outcome-weighted Variogram score* as

$$
    \mathrm{owVS}_{p}(F, \boldsymbol{y}; w) = w(\boldsymbol{y}) \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left( \frac{1}{\bar{w}_{F}} \mathbb{E} | X_{i} - X_{j} |^{p}w(\boldsymbol{X}) - | y_{i} - y_{j} |^{p} \right)^{2}.
$$

Since the Variogram score is a kernel score, the *threshold-weighted Variogram score* is defined as

$$
    \mathrm{twVS}_{p}(F, \boldsymbol{y}; v) = \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(\mathbb{E} | v(\boldsymbol{X})_{i} - v(\boldsymbol{X})_{j} |^{p} - | v(\boldsymbol{y})_{i} - v(\boldsymbol{y})_{j} |^{p} \right)^{2},
$$

where $v(\boldsymbol{y}) = (v(\boldsymbol{y})_{1}, \dots, v(\boldsymbol{y})_{d}) \in \Omega \subseteq \mathbb{R}^{d}$
for some chaining function $v : \Omega \to \Omega$ (Allen et al., 2024).

The *vertically-rescaled Variogram score* is

$$
    \mathrm{vrVS}_{p}(F, \boldsymbol{y}; w, x_{0}) = \mathbb{E} \left[ w(\boldsymbol{X})w(\boldsymbol{y}) \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| X_{i} - X_{j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2} \right] - \frac{1}{2} \mathbb{E} \left[ w(\boldsymbol{X})w(\boldsymbol{X^{\prime}}) \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| X_{i} - X_{j} |^{p} - | X_{i}^{\prime} - X_{j}^{\prime} |^{p} \right)^{2} \right] + \left( \mathbb{E} \left[ w(\boldsymbol{X}) \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| X_{i} - X_{j} |^{p} - | x_{0,i} - x_{0,j} |^{p} \right)^{2} \right] - w(\boldsymbol{y}) \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| y_{i} - y_{j} |^{p} - | x_{0,i} - x_{0,j} |^{p} \right)^{2} \right) \left( \mathbb{E}[w(\boldsymbol{X})] - w(\boldsymbol{y}) \right),
$$

for some $\boldsymbol{x}_{0} = (x_{0,1}, \dots, x_{0,d}) \in \mathbb{R}^{d}$. Analogously to the vertically
re-scaled energy score, the canonical choice is $\boldsymbol{x}_{0} = \boldsymbol{0}$,
unless the weight function is of the form $w(\boldsymbol{x}) = \mathbb{1}\{\boldsymbol{x} > \boldsymbol{t}\}$ or
$w(\boldsymbol{x}) = \mathbb{1}\{\boldsymbol{x} < t\}$, for some multivariate threshold
$\boldsymbol{t} \in \mathbb{R}$ (with $<$ and $>$ understood componentwise), in which case setting
$\boldsymbol{x}_{0} = \boldsymbol{t}$ recovers the threshold-weighted variogram score
(Allen et al., 2024, Proposition 4.10).


## Weight functions

The weighted scoring rules introduced above are all recovered when the weight function is
constant, e.g. $w(z) = 1$ for all $z \in \Omega$. Hence, weighted scoring rules increase the
flexibility offered to practitioners when evaluating forecasts.
The weight function can be chosen to direct the scoring rules to particular outcomes, so that
poor forecasts for these outcomes are penalised more heavily than poor forecasts for other
outcomes. In practice, it is often not trivial to decide which outcomes are "of interest",
and how much they should be emphasised when calculating the score; that is, what weight function
should be used. This decision is very application-specific and it is therefore difficult to provide
any general guidance regarding which weight function should be employed in practical applications.
Nonetheless, we list some common weight functions below.

### Univariate forecasts

Suppose $\Omega \subseteq \mathbb{R}$. The most common weight function used in practice is
an indicator weight function of the form

$$ w(z) = \mathbb{1}\{z > t\} \quad \text{or} \quad \mathbb{1}\{z < t\} $$

which restricts attention to outcomes above or below some relevant threshold $t \in \mathbb{R}$.
The threshold $t$ is typically chosen to be a fairly extreme quantile of previously observed
outcomes, though it could also correspond to a value used for policy making; a warning threshold,
for example.

More generally, we can define indicator weights on arbitrary regions of $\Omega$. For example,
we may be interested in values within a certain range,

$$ w(z) = \mathbb{1}\{a < z < b\} \quad \text{for some $-\infty \le a < b \le \infty$}, $$

which nests the above cases when $a = t$ and $b = \infty$ or when $a = -\infty$ and $b = t$;
or in values that either fall below a low threshold or exceed a high threshold,

$$ w(z) = \mathbb{1}\{z < a\} + \mathbb{1}\{z > b\} \quad \text{for some $-\infty \le a < b \le \infty$}. $$

Alternatively, Gneiting and Ranjan (2011) suggested using normal density and distribution functions
to define continuous weight functions that change smoothly over $\mathbb{R}$. Let $\phi_{\mu,\sigma}$
and $\Phi_{\mu, \sigma}$ denote the normal density and distribution functions respectively, with
location parameter $\mu$ and scale parameter $\sigma$. The weight function

$$ w(z) = \Phi_{\mu, \sigma}(z) $$

assigns a higher weight to higher outcomes, with the weight increasing gradually from 0 to 1.
This can be interpreted as a smoothed step function, where $\mu$ defines a threshold of interest,
and $\sigma$ determines the speed at which the weight changes from 0 before the threshold to 1 afterwards;
as $\sigma \to 0$, the weight function tends towards the indicator weight function $w(z) = \mathbb{1}\{z > \mu\}$,
while a larger value of $\sigma$ increases the weight more slowly. Similarly, the weight function

$$ w(z) = 1 - \Phi_{\mu, \sigma}(z) $$

assigns a higher weight to smaller outcomes, with analogous interpretation of $\mu$ and $\sigma$.

Using the normal density function allows central values to be emphasised,

$$ w(z) = \phi_{\mu, \sigma}(z). $$

In this case, $\sigma$ controls the concentration of the weight function around $\mu$, with the
weight function reducing to a point mass at $\mu$ as $\sigma \to 0$. Conversely, the lower and upper
tails can be targeted simultaneously using

$$ w(z) = 1 - \frac{\phi_{\mu, \sigma}(z)}{\phi_{\mu, \sigma}(\mu)}. $$

While these all employ the normal density and distribution functions, the normal distribution
could readily be replaced by any other location-scale family distribution. Allen (2024), for example,
consider the same weights defined using the logistic distribution.


### Multivariate forecasts

Multivariate weight functions (when $\Omega \subseteq \mathbb{R}^{d}$) can be defined similarly.
For example, it is common to employ an indicator weight function of the form

$$ w(\boldsymbol{z}) = \mathbb{1}\{z_{1} > t_{1}, \dots, z_{d} > t_{d}\} $$

to emphasise values above a threshold $t_{i} \in \mathbb{R}$ in each dimension. Different regions
of the outcome space can again be targeted by interchanging $<$ and $>$ signs. More generally,
this can be expressed as

$$ w(\boldsymbol{z}) = \mathbb{1}\{a_{1} < z_{1} < b_{1}, \dots, a_{d} < z_{d} < b_{d}\}, $$

for some $-\infty \le a_{i} < b_{i} \le \infty$ for $i = 1, \dots, d$.

Smooth weight functions can again be constructed using multivariate probability distributions.
For example, let $\phi_{\boldsymbol{\mu}, \Sigma}$ and $\Phi_{\boldsymbol{\mu}, \Sigma}$
denote the density and distribution functions of a multivariate normal distribution with
mean vector $\boldsymbol{\mu}$ and covariance matrix $\Sigma$. Then, high outcomes (in all dimensions)
could be targeted using a weight function of the form

$$ w(\boldsymbol{z}) = \Phi_{\boldsymbol{\mu}, \Sigma}(\boldsymbol{z}). $$

The interpretation of $\boldsymbol{\mu}$ and $\Sigma$ is similar to in the univariate case:
$\boldsymbol{\mu}$ can be thought of as a vector of thresholds, with $\Sigma$ controlling
the rate at which the weight function increases from 0 to 1 along each dimension. Low outcomes
can similarly be emphasised using

$$ w(\boldsymbol{z}) = 1 - \Phi_{\boldsymbol{\mu}, \Sigma}(\boldsymbol{z}), $$

while a high outcome in one dimension and a low outcome in another dimension could similarly
be targeted using appropriate manipulations of the multivariate normal distribution. For example,
when $d = 2$, the following weight function would target high values along the first dimension
and low values along the second,

$$ w(\boldsymbol{z}) = \mathrm{P}(X_{1} \le z_{1}, X_{2} > z_{2}), $$

where $\boldsymbol{X} = (X_{1}, X_{2})$ follows a multivariate normal distribution.

The multivariate normal density function can similarly be used to target central values,

$$ w(\boldsymbol{z}) = \phi_{\boldsymbol{\mu}, \Sigma}(\boldsymbol{z}), $$

while tails (in all dimensions) can be emphasised using

$$ w(\boldsymbol{z}) = 1 - \phi_{\boldsymbol{\mu}, \Sigma}(\boldsymbol{z})/\phi_{\boldsymbol{\mu}, \Sigma}(\boldsymbol{\mu}). $$

As in the univariate case, these are just particular examples, and by no means an exhaustive
list of the weight functions that should be employed in practice.


## Chaining functions

While most weighted scoring rules depend directly on a weight function, threshold-weighted
scoring rules can be defined in terms of a transformation function, or chaining function
$v : \Omega \to \Omega$. In the univariate case, a chaining function can be derived from a
weight function. However, in the multivariate case, there is generally no canonical way to
obtain a chaining function from a weight function. In the following, we discuss the choice
of chaining function in more detail.

### Univariate forecasts

Due to the two representations of the threshold-weighted CRPS, it is straightforward to map
a weight function to a chaining function in the univariate case ($\Omega \subseteq \mathbb{R}$).
In particular, given a weight function $w : \Omega \to [0, \infty)$, the chaining function is
simply an anti-derivative of the weight function. That is, the chaining function $v$ satisfied
$v(x) - v(x^{\prime}) = \int_{x^{\prime}}^{x} w(z) dz$ for all $x, x^{\prime} \in \Omega$.

When $w$ is constant, we recover the identity function $v(z) = z$, for all $z \in \Omega$,
and for all weight functions commonly used in practice, the chaining function is typically
straightforward to calculate. Several weight and chaining function combinations are listed
in Table 1 of Allen (2024).

For example, when $w(z) = \mathbb{1}\{z > t\}$, $t \in \mathbb{R}$, we get (up to an unimportant constant)

$$ v(z) = \max\{z, t\}, $$

and more generally for $w(z) = \mathbb{1}\{a < z < t\}$, $- \infty \le a < b \le \infty$ we have

$$ v(z) = \min\{\max\{z, a\}, b\}. $$

For $w(z) = \Phi_{\mu, \sigma}(z)$,

$$ v(z) = (z - \mu)\Phi_{\mu, \sigma}(z) + \sigma^{2} \phi_{\mu, \sigma}(z); $$

for $w(z) = 1 - \Phi_{\mu, \sigma}(z)$,

$$ v(z) = z - (z - \mu)\Phi_{\mu, \sigma}(z) - \sigma^{2} \phi_{\mu, \sigma}(z); $$

and for $w(z) = \phi_{\mu, \sigma}(z)$,

$$ v(z) = \Phi_{\mu, \sigma}(z). $$


### Multivariate forecasts

In the multivariate case $(\Omega \subseteq \mathbb{R}^{d})$, there is no default way to obtain
a chaining function given a multivariate weight function. One approach is to find the anti-derivative
of the weight function along each dimension separately. For example, for the weight function
$ w(\boldsymbol{z}) = \mathbb{1}\{a_{1} < z_{1} < b_{1}, \dots, a_{d} < z_{d} < b_{d}\}, $
a possible chaining function is

$$ v(\boldsymbol{z}) = (\min\{\max\{z_{1}, a_{1}\}, b_{1}\}, . . . , \min\{\max\{z_{d}, a_{d}\}, b_{d}\}). $$

In this case, the weight function represents a box in multivariate space, and $v$ projects
points not in the orthant onto its perimeter; the points inside the box, i.e., for which
the weight function is equal to one, remain unchanged.

Similarly, for the smooth weight functions based on multivariate Gaussian distribution and
density functions, a chaining function can be derived from a component-wise extension of
the chaining functions corresponding to univariate Gaussian weight functions. For example,
for $w(\boldsymbol{z}) = \Phi_{\boldsymbol{\mu}, \Sigma}(\boldsymbol{z})$, the chaining
function would become

$$ v(\boldsymbol{z}) = ((z_{1} - \mu_{1})\Phi_{\mu_{1}, \sigma_{1}}(z_{1}) + \sigma_{1}^{2} \phi_{\mu_{1}, \sigma_{1}}(z_{1}), \dots, (z_{d} - \mu_{d})\Phi_{\mu_{d}, \sigma_{d}}(z_{d}) + \sigma_{d}^{2} \phi_{\mu_{d}, \sigma_{d}}(z_{d})), $$

where $\sigma_{1}, \dots, \sigma_{d}$ are the standard deviations along each dimension.
This does not depend on the off-diagonal terms of the covariance matrix $\Sigma$, thereby
implicitly assuming that it is diagonal. However, when the weight function is
$w(\boldsymbol{z}) = \phi_{\boldsymbol{\mu}, \Sigma}(\boldsymbol{z})$, we can readily
use

$$ v(\boldsymbol{z}) = \Phi_{\boldsymbol{\mu}, \Sigma}(\boldsymbol{z}). $$

A more detailed discussion on multivariate chaining functions is available in Allen et al. (2023).
