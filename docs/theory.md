# The theory of proper scoring rules

## Definitions

Suppose we issue a probabilistic forecast $F$ for an outcome variable $Y$ that takes values
in some set $\Omega$. A probabilistic forecast for $Y$ is a probability distribution over
$\Omega$.

A *scoring rule* is a function

$$ S : \mathcal{F} \times \Omega \to \overline{\mathbb{R}}, $$

where $\mathcal{F}$ denotes a set of probability distributions over $\Omega$, and
$\overline{\mathbb{R}}$ denotes the extended real line. A scoring rule therefore takes a
probabilistic forecast $F \in \mathcal{F}$ and a corresponding outcome $y \in \Omega$ as
inputs, and outputs a real (possibly infinite) value, or score, $S(F, y)$. This score quantifies the
accuracy of $F$ given that $y$ has occurred.

We assume that scoring rules are negatively oriented, so that a lower score indicates
a more accurate forecast, though the definitions and examples discussed below can easily
be modified to treat scoring rules that are positively oriented.

It is widely accepted that scoring rules should be *proper*. A scoring rule $S$ is proper
(with respect to $\mathcal{F}$) if, when $Y \sim G$,

$$
    \mathbb{E} S(G, Y) \leq \mathbb{E} S(F, Y) \quad \text{for all $F, G \in \mathcal{F}$}.
$$

That is, the score is minimised in expectation by the true distribution underlying the outcome $Y$.
Put differently, if we believe that the outcome arises according to distribution $G$, then
our expected score is minimised by issuing $G$ as our forecast; proper scoring rules therefore
encourage honest predictions, and discourage hedging. A scoring rule is *strictly proper*
if the above inequality holds with equality if and only if $F = G$.


## Examples

### Binary outcomes:

Suppose $\Omega = \{0, 1\}$, so that $Y$ is a binary outcome that either occurs $(Y = 1)$
or does not $(Y = 0)$. A probabilistic forecast for $Y$ is a single value $F \in [0, 1]$
that quantifies the probability that $Y = 1$. To evaluate such a forecast, popular scoring
rules include the *Brier score* and *Logarithmic (Log) score*.

The Brier score is defined as

$$
    \mathrm{BS}(F, y) = (F - y)^{2},
$$

while the Log score is defined as

$$
    \mathrm{LS}(F, y) = -\log |F + y - 1| =
    \begin{cases}
        -\log F & \text{if} \quad y = 1, \\
        -\log (1 - F) & \text{if} \quad y = 0.
    \end{cases}
$$

Other popular binary scoring rules include the spherical score, power score, and pseudo-spherical
score.


### Categorical outcomes:

Suppose now that $Y$ is a categorical variable, taking values in $\Omega = \{1, \dots, K\}$,
for $K > 1$. Binary outcomes constitute a particular case when $K = 2$.
A probabilistic forecast for $Y$ is a vector

$$
    F = (F_{1}, \dots, F_{K}) \in [0, 1]^{K} \quad \text{such that} \quad \sum_{i=1}^{K} F_{i} = 1.
$$

The $i$-th element of $F$ represents the probability that $Y = i$, for $i = 1, \dots, K$.

Proper scoring rules for binary outcomes can readily be used to evaluate probabilistic
forecasts for categorical outcomes, by applying the score separately for each category,
and then summing these $K$ scores.

For example, the Brier score becomes

$$
    \mathrm{BS}_{K}(F, y) = \sum_{i=1}^{K} (F_{i} - \mathbf{1}\{y = i\})^{2}.
$$

where $\mathbf{1}\{ \cdotp \}$ denotes the indicator function.
When $K = 2$, we recover the Brier score for binary outcomes (multiplied by a factor of 2).

The Log score can similarly be extended, though it is more common to define the Log score for
categorical outcomes as

$$
    \mathrm{LS}(F, y) = -\log F_{y}.
$$

As in the binary case, this Log score evaluates the forecast $F$ only via the probability
assigned to the outcome that occurs.

The Brier score assesses the forecast probability separately at each category. However, if the
categories are ordered, i.e. the outcome is ordinal, then one could argue that the distance
between categories should also be considered. For example, if $K = 3$ and $y = 3$, then
the forecast $F^{(1)} = (1, 0, 0)$ would receive the same score as $F^{(2)} = (0, 1, 0)$,
according to both the Brier score and the Log score. But one could argue that the second forecast
$F^{(2)}$ should be preferred: $F^{(1)}$ assigns all probability to the first category,
$F^{(2)}$ assigns all probability to the second category, and the second category is closer to the third.

To account for the ordering of the categories, it is common to apply the Brier score and
Log score to cumulative probabilities. Let

$$
\begin{align}
    \tilde{F} &= (\tilde{F}_{1}, \dots, \tilde{F}_{K}) \in [0, 1]^{K} \quad \text{with
    $\tilde{F}_{j} = \sum_{i=1}^{j} F_{i}$} \quad &\text{for $j = 1, \dots, K$,} \\
    \tilde{y} &= (\tilde{y}_{1}, \dots, \tilde{y}_{K}) \in \{0, 1\}^{K} \quad \text{with
    $\tilde{y}_{j} = \sum_{i=1}^{j} \mathbf{1}\{y = i\}$} \quad &\text{for $j = 1, \dots, K$.}
\end{align}
$$

Then, the *Ranked Probability Score (RPS)* is defined as

$$
    \mathrm{RPS}(F, y) = \sum_{i=1}^{K} (\tilde{F}_{j} - \tilde{y}_{j})^{2},
$$

and the *Ranked Logarithmic Score (RLS)* is

$$
    \mathrm{RLS}(F, y) = - \sum_{i=1}^{K} \log | \tilde{F}_{j} + \tilde{y}_{j} - 1|.
$$

These categorical scoring rules can also be implemented when $K = \infty$,
as is the case for unbounded count data, for example. Other scoring rules for binary
outcomes can similarly be extended in this way to construct scoring rules for categorical
outcomes.


### Continuous outcomes:

Let $Y \in \mathbb{R}$ and suppose $F$ is a cumulative distribution function over the real line.

We can similarly define proper scoring rules for continuous outcomes in terms of scoring rules
for binary outcomes.

The *Continuous Ranked Probability Score (CRPS)* is defined as

$$
\begin{align*}
    \mathrm{CRPS}(F, y) &= \int_{-\infty}^{\infty} (F(x) - \mathbf{1}\{y \le x\})^{2} dx \\
    &= \mathbb{E} | X - y | - \frac{1}{2} \mathbb{E} | X - X^{\prime} |,
\end{align*}
$$

where $X, X^{\prime} \sim F$ are independent.
The CRPS is defined as the Brier score for threshold exceedance forecasts integrated over all
thresholds. This corresponds to a generalisation of the RPS for when there are (uncountably)
infinite possible categories.

Similarly, the *Continuous Ranked Logarithmic Score (CRLS)* is defined as

$$
    \mathrm{CRLS}(F, y) = -\int_{-\infty}^{\infty} \log |F(x) + \mathbf{1}\{y \le x\} - 1| dx.
$$

The Log score can additionally be generalised to continuous outcomes whenever
the forecast $F$ has a density function, denoted here by $f$. In this case, the
Log score is defined as

$$
    \mathrm{LS}(F, y) = - \log f(y).
$$

The Log score again depends only on the forecast distribution $F$ at the observation $y$,
ignoring the probability density assigned to other outcomes. This scoring rule is therefore
*local*.

When $F$ is a normal distribution, the Log score simplifies to the *Dawid-Sebastiani* score,

$$
    \mathrm{DS}(F, y) = \frac{(y - \mu_{F})^{2}}{\sigma_{F}^{2}} + 2 \log \sigma_{F},
$$

where $\mu_{F}$ and $\sigma_{F}$ represent the mean and standard deviation of the forecast
distribution $F$. While the Dawid-Sebastiani score corresponds to the Log score for a normal distribution,
it also constitutes a proper scoring rule for any other forecast distribution with finite
mean and variance. The Dawid-Sebastiani score evaluates forecasts only via their mean and standard
deviation, making it easy to implement in practice, but insensitive to higher moments of the
predictive distribution.


### Multivariate outcomes:

Let $\boldsymbol{Y} \in \mathbb{R}^{d}$, with $d > 1$, and suppose $F$ is a multivariate probability distribution.

If $F$ admits a multivariate density function $f$, then the Log score can be defined analogously to
in the univariate case,

$$
    \mathrm{LS}(F, \boldsymbol{y}) = - \log f(\boldsymbol{y}).
$$

The Dawid-Sebastiani score can similarly be extended to higher dimensions
by replacing the predictive mean $\mu_{F}$ and variance $\sigma_{F}^{2}$ with the mean vector
$\boldsymbol{\mu}_{F} \in \mathbb{R}^{d}$ and covariance matrix $\Sigma_{F} \in \mathbb{R}^{d \times d}$.
This becomes

$$
    \mathrm{DS}(F, \boldsymbol{y}) = (\boldsymbol{y} - \boldsymbol{\mu}_{F})^{\top} \Sigma_{F}^{-1} (\boldsymbol{y} - \boldsymbol{\mu}_{F}) + \log \det(\Sigma_{F}),
$$

where $\top$ denotes the vector transpose, and $\det$ the matrix determinant. The Dawid-Sebastiani
score is equivalent to the Log score for a multivariate normal distribution. However, the
Dawid-Sebastiani score is more readily applicable than the Log score, since it depends only on
the predictive mean vector and covariance matrix, and does not require a predictive density.
However, particularly in high dimensions, the predictive covariance matrix is often not available,
or must be estimated from a finite sample.

Instead, it is common to evaluate multivariate forecasts using the *Energy score*,

$$
    \mathrm{ES}(F, \boldsymbol{y}) = \mathbb{E} \| \boldsymbol{X} - \boldsymbol{y} \| - \frac{1}{2} \mathbb{E} \| \boldsymbol{X} - \boldsymbol{X}^{\prime} \|,
$$

where $\boldsymbol{X}, \boldsymbol{X}^{\prime} \sim F$ are independent, and $\| \cdot \|$ is the Euclidean distance on $\mathbb{R}^{d}$.
The Energy score can be interpreted as a multivariate generalisation of the CRPS, which is recovered when $d = 1$.

The Energy score is sensitive to both the univariate performance of the multivariate forecast distribution
along each margin, as well as the predicted dependence structure between the different dimensions.
The *Variogram score* was introduced as an alternative scoring rule that is less sensitive to the
univariate forecast performance, thereby focusing on the multivariate dependence structure.
The Variogram score is defined as

$$
    \mathrm{VS}(F, \boldsymbol{y}) = \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left( \mathbb{E} | X_{i} - X_{j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2},
$$

where $\boldsymbol{X} \sim F$, and $h_{i,j} \ge 0$ are weights assigned to different pairs of dimensions.
The Variogram score therefore measures the distance between between the expected variogram of
the forecast distribution, and the variogram of the single observation $\boldsymbol{y}$.


## Characterisations

The proper scoring rules introduced above are particular examples that are
commonly used to evaluate probabilistic forecasts in practice. More generally, it is possible
to derive characterisations of all proper scoring rules, allowing users to easily construct
novel scoring rules that incorporate their personal preferences.


### Schervish representation

Schervish (1969) demonstrated that all proper scoring rules for binary outcomes can be
written in the following form (up to unimportant technical details):

$$
    S(F, y) = \int_{(0, 1)} c 1\{F > c, y = 0\} + (1 - c) 1\{F < c, y = 1\} + c(1 - c) 1\{F = c\} d \nu(c),
$$

for some non-negative measure $\nu$ on $(0, 1)$. The measure $\nu$ emphasises particular trade-offs
between over-prediction ($F$ high, $y = 0$) and under-prediction ($F$ low, $y = 1$).

The Brier score corresponds to the measure $d \nu(c) = 2 dc$, and the Log score $d \nu(c) = \frac{1}{c(1 - c)}dc$.

This essentially means that any proper scoring rule for binary outcomes can be obtained by choosing a weight function
over the trade-off parameter $c$, and plugging this into the characterisation above. If $\nu$ is a
Dirac measure at a particular value of $c \in (0, 1)$, then we recover the (asymmetric) *zero-one score*.
Every proper scoring rule can therefore be expressed as a weighted integral of the zero-one score over all
trade-off parameters.


### Convex functions

More generally, for categorical forecasts, a scoring rule is proper if and only if there exists a convex function
$g : [0, 1]^{K} \to \mathbb{R}$ such that

$$
    S(F, y) = \langle g^{\prime}(F), F \rangle - g(F) - g^{\prime}(F)_{y} \quad \text{for all $F \in [0, 1]^{K}$ and $y \in \{1, \dots, K\}$,}
$$

where $g^{\prime}(F) \in \mathbb{R}^{K}$ is a sub-gradient of $g$ at $F \in \mathcal{F}$,
and $g^{\prime}(F)_{y}$ is the $y$-th element of this vector, for $y \in \{1, \dots, K\}$.

If $K = 2$ and $g$ is smooth, then the Schervish representation is recovered by setting
$d\nu(c) = g^{\prime \prime}(c) dc$. That is, the Lesbesgue density of $\nu$ is $g^{\prime \prime}$.
The Brier score corresponds to the convex function $g(F) = \sum_{j=1}^{K} F_{j}^{2} - 1$,
and the Log score to the Shannon entropy function $g(F) = \sum_{j=1}^{K} F_{j} \log F_{j}$.

Gneiting and Raftery (2007) generalised this to arbitrary outcome domains by showing that
a scoring rule $S$ is proper relative to the set $\mathcal{F}$ if and only if there exists a
convex function $g : \Omega \to \mathbb{R}$ such that

$$
    S(F, y) = \int g^{\prime}(F, z) dF(z) - g(F) - g^{\prime}(F, y) \quad \text{for all $F \in \mathcal{F}$ and $y \in \Omega$,}
$$

where $g^{\prime}(F, \cdot)$ is a subtangent of $g$ at $F$ (Gneiting and Raftery, 2007, Theorem 1).
The class of strictly proper scoring rules with respect to $\mathcal{F}$ is characterised by
replacing convex with strictly convex.

For example, the CRPS corresponds to the convex function $g(F) = -\mathbb{E} |X - X^{\prime}|$
for $X, X^{\prime} \sim F$ independent. The energy score corresponds to the same function, with the
absolute distance replaced with the Euclidean distance on $\mathbb{R}^{d}$.


## Classes of scoring rules

### Local scoring rules

Local scoring rules assume that forecasts should be evaluated based only on what has occurred,
and not what could have occurred but did not. Hence, local scoring rules depend on the forecast
distribution only via the predictive density evaluated at the outcome $y$. More generally,
assuming the forecast distribution $F$ admits a density function $f$ whose derivatives
$f^{\prime}, f^{\prime \prime}, \dots, f^{(j)}$ exist, a scoring rule $S$ is called *local of order $j$*
if there exists a function $s : \mathbb{R}^{2+j} \to \overline{\mathbb{R}}$ such that

$$
    S(F, y) = s(y, f(y), f^{\prime}(y), \dots, f^{(j)}(y)).
$$

That is, the scoring rule $S$ can be written as a function of $y$ and the first $j$ derivatives
of the forecast distribution evaluated at $y$.

The Log score is the only proper scoring rule that is local of order $j = 0$ (up to equivalence),
see Bernardo (1979). However, local proper scoring rules of higher orders also exist, the most
well-known example being the *Hyv&auml;rinen score*, which is local of order $j = 2$.


### Matheson and Winkler representation

Scoring rules that are not local are often called *distance-sensitive*, since they take into
account the distance between the forecast distribution and the observation.
One general approach to construct distance-sensitive proper scoring rules for univariate real-valued outcomes
is to use a proper scoring rule for binary outcomes to evaluate threshold exceedance forecasts,
and to integrate this over all thresholds.

That is, if $S_{0}$ is a proper scoring rule for binary outcomes, then

$$
    S(F, y) = \int_{-\infty}^{\infty} S_{0}(F(x), \mathbf{1}\{y \leq x\}) dH(x),
$$

is a proper scoring rule for univariate real-valued outcomes, for some non-negative measure $H$ on the real line.

The CRPS is equivalent to the case when $S_{0}$ is the Brier score and $H$ is the Lebsegue measure,
i.e. $dH(x) = dx$. Replacing the Brier score with the Log score recovers the Continuous Ranked
Logarithmic Score (CRLS). By changing the measure $H$ in the above construction, different regions of
the outcome space can be assigned different weight, thereby allowing users to emphasise particular outcomes
during evaluation. This provides a means to construct weighted scoring rules (see below).

This construction can also be used to evaluate multivariate probabilistic forecasts, by replacing $F(x)$
with the multivariate distribution function, and $\mathbf{1}\{ y \leq x\}$ with
$\mathbf{1}\{ y_{1} \leq x_{1}, \dots, y_{d} \leq x_{d}\}$, before integrating over all
$\boldsymbol{x} \in \mathbb{R}^{d}$. Gneiting and Raftery (2007) use this to introduce a multivariate
CRPS.



### Kernel scores

Many popular scoring rules belong to the very general class of *kernel scores*, which are
scoring rules defined in terms of positive definite kernels. A positive definite kernel on
$\Omega$ is a symmetric function $k : \Omega \times \Omega \to \mathbb{R}$, such that

$$
    \sum_{i=1}^{n} \sum_{j=1}^{n} a_{i} a_{j} k(x_{i}, x_{j}) \geq 0,
$$

for all $n \in \mathbb{N}$, $a_{i}, a_{j} \in \mathbb{R}$, and $x_{i}, x_{j} \in \Omega$.
A positive definite kernel constitutes an inner product in a feature space, and can therefore
be loosely interpreted as a measure of similarity between its two inputs.

The *kernel score* corresponding to the positive definite kernel $k$ is defined as

$$
    S_{k}(F, y) = \frac{1}{2} \mathbb{E} k(y, y) + \frac{1}{2} \mathbb{E} k(X, X^{\prime}) - \mathbb{E} k(X, y),
$$

where $X, X^{\prime} \sim F$ are independent. The first term on the right-hand-side does not
depend on $F$, so could be removed, but is included here to ensure that the kernel score is always
non-negative, and to retain an interpretation in terms of Maximum Mean Discrepancies (see below).

The energy score is the kernel score associated with any kernel

$$
    k(x, x^{\prime}) = \| x - x_{0} \| + \| x^{\prime} - x_{0} \| - \| x - x^{\prime} \|
$$

for arbitrary $x_{0} \in \Omega$, which encompasses the CRPS when $d = 1$. We
refer to these kernels as *energy kernels*.

Another popular kernel is the Gaussian kernel,

$$
    k_{\gamma}(x, x^{\prime}) = \exp \left( - \gamma \| x - x^{\prime} \|^{2} \right)
$$

for some length-scale parameter $\gamma > 0$. Plugging this into the kernel score definition
above yields the *Gaussian kernel score*.

Other examples include the Brier score, variogram score, and angular CRPS.
Allen et al. (2023) demonstrate how changing the kernel can be used to target particular outcomes,
and illustrate that the threshold-weighted CRPS constitutes a particular sub-class of kernel scores.
Any positive definite kernel can be used to define a scoring rule. This allows for very
flexible forecast evaluation on arbitrary outcome domains.

Steinwart and Ziegel (2021) discuss the connection between kernel scores and maximum mean
discrepancies (MMDs), which are commonly used to measure the distance between probability
distributions. Aronszajn (195) demonstrated that every positive definite kernel $k$ induces
a Hilbert space of functions, referred to as a Reproducing Kernel Hilbert Space and denoted by $\mathcal{H}_{k}$.
A probability distribution $F$ on $\Omega$ can be converted to an element in an RKHS via its kernel mean embedding,

$$
\mu_{F} = \int_{\Omega} k(x, \cdotp)  dF(x),
$$

allowing kernel methods to be applied to probabilistic forecasts.

For example, to calculate the distance between two distributions $F$ and $G$, we can calculate
the RKHS norm $\| \cdot \|_{\mathcal{H}_{k}}$ betewen their kernel mean embeddings,
$\| \mu_{F} - \mu_{G} \|_{\mathcal{H}_{k}}$. This is called the *Maximum Mean Discrepancy (MMD)*
between $F$ and $G$. In practice, it is common to calculate the squared MMD, which can be expressed as

$$
    \| \mu_{F} - \mu_{G} \|_{\mathcal{H}_{k}}^{2} = \mathbb{E} k(Y, Y^{\prime}) + \mathbb{E} k(X, X^{\prime}) - 2\mathbb{E} k(X, Y),
$$

where $X, X^{\prime} \sim F$ and $Y, Y^{\prime} \sim G$ are independent.

The kernel score corresponding to $k$ is simply (half) the squared MMD between the probabilistic
prediction $F$ and a point measure at the observation $y$, denoted $\delta_{y}$.
The propriety of kernel scores follows from the non-negativeness of the MMD (Steinwart and Ziegel, 2021).
This relationship between kernel scores and MMDs is leveraged by Allen et al. (2024) to
introduce efficient methods when pooling probabilistic forecasts.



### Weighted scoring rules

Often, some outcomes lead to larger impacts than others, making accurate forecasts for these outcomes
more valuable. To account for this during forecast evaluation, scoring rules could be adapted so that
they assign more weight to outcomes that are higher impact. Weighted scoring rules extend conventional
scoring rules by incorporating non-negative weight functions $w : \Omega \to [0, \infty)$ that control
how much emphasis is to be placed on each outcome in $\Omega$.

For example, for categorical outcomes, the Brier score and RPS can easily be extended by weighting each
term of the summation in their definition, thereby assigning more weight to particular categories.
For continuous outcomes, this can be extended to the CRPS by incorporating a weight function into
the integral, yielding the *threshold-weighted CRPS*

\begin{align*}
    \mathrm{twCRPS}(F, y; w) &= \int_{\mathbb{R}} (F(x) - \mathbb{1}\{y \leq x\})^{2} w(x) dx \\
    &= \mathbb{E} | v(X) - v(y) | - \frac{1}{2} \mathbb{E} | v(X) - v(X^{\prime}) |,
\end{align*}

where $X, X^{\prime} \sim F$ are independent, and $v$ is such that $v(x) - v(x^{\prime}) = \int_{x^{\prime}}^{x} w(z) dz$
for any $x, x^{\prime} \in \mathbb{R}$. Allen et al. (2022) refer to $v$ as the *chaining function*.
The threshold-weighted CRPS corresponds to choosing the measure $H$ in the construction in the previous section to
have Lebesgue density $w$. This can trivially be applied to the CRLS to yield a threshold-weighted CRLS,
as well as any scoring rule defined using the construction above.

Alternatively, the second expression highlights that the threshold-weighted CRPS is a kernel score,
and that the weighting transforms the forecasts, according to the chaining function $v$, which is an
anti-derivative of the weight function $w$.

For example, a popular weight function is $w(z) = \mathbb{1}\{z > t\}$, which restricts attention to
values above some threshold of interest $t \in \mathbb{R}$. In this case, a possible chaining function
$v$ is $v(z) = \max(z, t)$, in which case the threshold-weighted CRPS censors the observation and the
forecast distribution at the threshold $t$, and then calculates the standard CRPS for the censored forecast and
observation.

Evaluating censored forecast distributions was also proposed by Diks et al. (2011) when introducing
weighted versions of the Log score. They introduce the *censored likelihood score* as

$$
    \mathrm{ceLS}(F, y; w) = - w(y) \log f(y) - (1 - w(y)) \log \left( 1 - \int_{\mathbb{R}} w(z) f(z) dz \right).
$$

Rather than censoring the distribution, Diks et al. (2011) additionally propose the *conditional likelihood score*,
which evaluates the conditional distribution given the weight function,

$$
    \mathrm{coLS}(F, y; w) = - w(y) \log f(y) + w(y) \log \left( \int_{\mathbb{R}} w(z) f(z) dz \right).
$$



### Consistent scoring functions

When only a functional of the forecast distribution $F$ is of interest,
proper scoring rules can be defined using consistent scoring functions.

Throughout, we have considered the case when the forecast $F$ is probabilistic. Often,
however, for reasons of communication or tradition, the forecast $F$ is not probabilistic.
For example, it is common to report the expected outcome, or a conditional quantile.

These forecasts can be evaluated using *scoring functions*. Scoring functions are similar
in essence to scoring rules, but they take a point-valued rather than probabilistic forecast
as input,

$$
    s : \Omega \times \Omega \to [0, \infty].
$$

We use a lower case $s$ to distinguish scoring functions from scoring rules.

To impose theoretical guarantees upon the scoring function, it is necessary to assume that
the point-valued forecast comes from an underlying probability distribution. That is,
the forecast is a functional $\text{T}$ of a predictive distribution. For example, $\text{T}$ could
be the mean, median, or quantile. A scoring rule is called *consistent* for the functional $\text{T}$
(relative to a class of distributions $\mathcal{F}$), if, when $Y \sim G$,

$$
    \mathbb{E} s(x_{G}, Y) \leq \mathbb{E} s(x, Y) \quad \text{for all $x \in \Omega, G \in \mathcal{F}$,}
$$

where $x_{G} \in \text{T}(G)$.

For example, the *squared error*

$$
    s(x, y) = (x - y)^{2}
$$

is consistent for the mean functional, and the *quantile score* (also called *pinball loss* or *check loss*)

$$
    s_{\alpha}(x, y) = (\mathbf{1}\{y \leq x\} - \alpha)(x - y) =
    \begin{cases}
        (1 - \alpha)|x - y| & \quad \text{if $y \leq x$}, \\
        \phantom{(1 - }\alpha \phantom{)}|x - y| & \quad \text{if $y \geq x$}
    \end{cases}
$$

is consistent for the $\alpha$-quantile (for $\alpha \in (0, 1)$). When $\alpha = 0.5$,
we recover the *absolute error*

$$
    s(x, y) = |x - y|,
$$

which is consistent for the median functional. Note that these are not the only scoring rules
that are consistent for the mean, median, and quantiles (see Gneiting 2010, Ehm et al., 2016).

Proper scoring rules can be constructed using consistent scoring functions. In particular,
if $s$ is a consistent scoring function for a functional $\text{T}$ (relative to $\mathcal{F}$), then

$$
    S(F, y) = s(\text{T}(F), y)
$$

is a proper scoring rule (relative to $\mathcal{F}$). Hence, the squared error, quantile loss,
and absolute error above all induce proper scoring rules.
These scoring rules evaluate $F$ only via the functional of interest.
Hence, while they are proper, they are generally not strictly proper.

This framework can additionally be used to evaluate interval forecasts, or prediction intervals.
Given a central interval forecast in the form of a lower and upper value $L, U \in \mathbb{R}$ with $L < U$,
the interval score is defined as

$$
    \mathrm{IS}([L, U], y) = |U - L| + \frac{2}{\alpha} (y - u) \mathbf{1} \{ y > u \} + \frac{2}{\alpha} (l - y) \mathbf{1} \{ y < l \}.
$$
