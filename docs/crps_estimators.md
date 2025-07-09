(crps-estimators)=
# Ensemble forecasts

Suppose the forecast is an *ensemble forecast*. That is, the forecast is
available via $M$ values, $x_{1}, \dots, x_{M} \in \Omega$, called *ensemble members*.
Each ensemble member can be interpreted as a possible outcome, so that the distribution of the
members represents the range of outcomes that could occur.
Ensemble forecasts can be generated, for example, from generative machine learning models,
Markov chain Monte Carlo (MCMC) methods, or from the output of physical weather or climate models.

Note that in some fields, the term ensemble forecast refers to a (linear) aggregation of different
forecasts. This differs from the nomenclature employed here, where we instead refer to an
ensemble forecast as a discrete predictive distribution defined by the empirical distribution
of the values $x_{1}, \dots, x_{M}$. For example, when $\Omega \subseteq \mathbb{R}$,
the ensemble forecast is

$$
F_{M}(x) = \frac{1}{M} \sum_{i=1}^{M} \mathbb{1}\{x_{i} \le x\},
$$

where $\mathbb{1}\{\cdotp\}$ is the indicator function.

This forecast can be evaluated by plugging $F_{M}$ into the definition of scoring rules.
Since this empirical distribution does not emit a density function, ensemble forecasts can
generally not be evaluated using the Log score without making additional assumptions about
the shape of the distribution at unobserved values.

In the following, we discuss different
ways to calculate the Continuous Ranked Probability Score (CRPS) when the forecast is an
ensemble forecast, and also how the interpretation of the ensemble members changes the way
scores should be calculated; in particular, when a *fair* scoring rule should be employed.


## CRPS calculation

Suppose $\Omega \subseteq \mathbb{R}$, so that $F$ is a predictive distribution function.
The CRPS can be expressed in several ways:

$$
\begin{align}
\mathrm{CRPS}(F, y) &= \int_{-\infty}^{\infty} (F(z) - \mathbb{1}\{y \le z\})^{2} dz \\
&= \int_{0}^{1} (\mathbb{1}\{F^{-1}(\alpha) \ge y\} - \alpha)(F^{-1}(\alpha) - y) d\alpha \\
&= \mathbb{E}|X - y| - \frac{1}{2} \mathbb{E}|X - X^{\prime}|, \\
\end{align}
$$

where $F^{-1}$ is the generalised inverse of $F$, and $X, X^{\prime} \sim F$ are independent
(Matheson and Winkler, 1976; Laio and Tamea, 2007; Gneiting and Raftery, 2007).

The different representations of the CRPS lead to many different ways to calculate the score.
While these all yield the same score, the different representations can often be calculated with
different computational complexity, meaning some are faster to implement than others. This is particularly
relevant when the forecast is an ensemble.
Some common options to calculate the CRPS for ensemble forecasts are given below.


### Energy form (NRG)

Grimit et al. (2006) showed that the CRPS of the empirical distribution $F_{M}$ is equal to

$$
\mathrm{CRPS}_{\text{NRG}}(F_{M}, y) = \frac{1}{M} \sum_{i=1}^{M}|x_i - y| - \frac{1}{2 M^2}\sum_{i=1}^{M} \sum_{j=1}^{M}|x_i - x_j|,
$$

which follows from the kernel score representation of the CRPS given by Gneiting and Raftery (2007).
This has been referred to as the *energy form* of the CRPS, and can be calculated with $O(M^2)$ complexity.


### Quantile decomposition form (QD)

Jordan (2016) introduced a *quantile decomposition form* of the CRPS,

$$
\mathrm{CRPS}_{\mathrm{QD}}(F_{M}, y) = \frac{2}{M} \sum_{i=1}^{M}\left[\mathbb{1}\{y \le x_{(i)}\} - \frac{2i - 1}{2M} \right] (x_{(i)} - y),
$$

where $x_{(i)}$ denotes the $i$-th order statistic of the ensemble members,
so that $x_{(1)} \le x_{(2)} \le \dots \le x_{(M)}$. This can be calculated
with $O(M\cdot\mathrm{log}M)$ complexity, including the sorting of the ensemble members.


### Probability weighted moment form (PWM)

Taillardat et al. (2016) proposed an expression of the CRPS based on probability weighted moments,

$$
\mathrm{CRPS}_{\mathrm{PWM}}(F_{M}, y) = \frac{1}{M} \sum_{i=1}^{M}|x_{(i)} - y| + \frac{M - 1}{M} \left(\hat{\beta_0} - 2\hat{\beta_1}\right),
$$

where $\hat{\beta_0} = \frac{1}{M} \sum_{i=1}^{M}x_{(i)}$ and $\hat{\beta_1} = \frac{1}{M(M-1)} \sum_{i=1}^{M}(i - 1)x_{(i)}$.
This *probability weighted moment form* of the CRPS also requires sorting the ensemble,
and in total runs with $O(M\cdot\mathrm{log}M)$ complexity.

The above expression differs slightly from that initially introduced by Taillardat et al. (2016), in that
$\hat{\beta_{0}}$ and $\hat{\beta_{1}}$ have been scaled by $(M - 1)/M$. This ensures that the CRPS is
equivalent to the other forms listed in this section. Without rescaling by $(M - 1)/M$,
we recover the *fair* version of the CRPS, which is discussed in more detail in the following
sections.


### Integral form (INT)

Alternatively, the *integral form* of the CRPS,

$$
\mathrm{CRPS}_{\text{INT}}(F_{M}, y) = \int_{\mathbb{R}} \left[ \frac{1}{M}
    \sum_{i=1}^M \mathbb{1}\{x_i \le x \} - \mathbb{1}\{y \le x\}  \right] ^2 dx,
$$

can be numerically approximated. This also runs with $O(M\cdot\mathrm{log}M)$ complexity,
but introduces small errors due to the numerical approximation.


## Fair scoring rules

The above expressions of the CRPS assume that we wish to assess the empirical distribution defined by
the ensemble members $x_{1}, \dots, x_{M}$. However, one could argue that the ensemble forecast
should not be treated as a forecast distribution in itself, but rather a random sample of values from
an underlying (unknown) distribution, say $F$. We may wish to evaluate this underlying distribution
rather than the empirical distribution of the available sample ($F_{M}$).

The choice to interpret $x_{1}, \dots, x_{M}$ as a sample or as a distribution in itself is
not trivial. For example, by treating the ensemble forecast as an empirical distribution, we implicitly
assume that the forecast probability that the outcome exceeds all ensemble members is zero,
which may not be how we interpret the ensemble. However, if the forecast is to be used for
decision making, then we generally have to make decisions based only on the information available
from $x_{1}, \dots, x_{M}$, and not the underlying, unknown distribution. For this reason,
`scoringrules` treats the ensemble forecast as an empirical distribution by default.

If we wish to evaluate the underlying distribution $F$ rather than the empirical distribution
defined by the ensemble members, we must adapt the scoring rule to account for the fact that
$x_{1}, \dots, x_{M}$ are only a selection of possible values that could have been drawn from $F$.
For this purpose, Ferro (2014) introduced *fair* scoring rules.

A scoring rule for ensemble forecasts is fair if the expected score is minimised when the ensemble members
are random draws from the distribution of $Y$. That is, given a class of probability distributions
$\mathcal{F}$ on $\Omega$, a scoring rule $S$ is fair (with respect to $\mathcal{F}$) if,
when $Y \sim G \in \mathcal{F}$,

$$
\mathbb{E}S(G_{M}, Y) \le \mathbb{E}S(F_{M}, Y),
$$

for all $F, G \in \mathcal{F}$  , where $F_{M}$ represents a (random) sample of $M$ independent values
drawn from $F$, i.e. $x_{1}, \dots, x_{M} \sim F$. Note that the expectation is taken over both
the outcome $Y$ and the sample values.

Given a proper scoring rule $S$, a fair version of the score (denoted $S^{f}$) can be attained
by ensuring that $\mathbb{E}S(F_{M}, Y) = \mathbb{E}S(F, Y)$, for all $F \in \mathcal{F}$,
in which case the score assigned to $F_{M}$ represents an unbiased estimator of the score
assigned to the underlying distribution $F$. Some examples of fair versions of popular
scoring rules are provided below.


### Kernel scores

A very general class of scoring rules is the class of kernel scores.
Recall that the *kernel score* corresponding to a positive definite kernel $k$ on $\Omega$ is defined as

$$
    S_{k}(F, y) = \frac{1}{2} k(y, y) + \frac{1}{2} \mathbb{E} k(X, X^{\prime}) - \mathbb{E} k(X, y),
$$

where $X, X^{\prime} \sim F$ are independent.

Since kernel scores are defined in terms of expectations, they can be calculated easily
for ensemble forecasts by replacing the expectations with empirical means over the ensemble members.
That is,

$$
    S_{k}(F_{M}, y) = \frac{1}{2} k(y, y) + \frac{1}{2 M^{2}} \sum_{i=1}^{M} \sum_{j=1}^{M} k(x_{i}, x_{j}) - \frac{1}{M} \sum_{i=1}^{M} k(x_{i}, y).
$$

Many kernel scores used in practice are constructed using a kernel of the form

$$ k(x, x^{\prime}) = \rho(x, x_{0}) + \rho(x^{\prime}, x_{0}) - \rho(x, x^{\prime}) $$

where $x_{0} \in \Omega$, and $\rho : \Omega \to \Omega \to [0, \infty)$ is a symmetric,
conditionally negative definite function with $\rho(x, x) = 0$ for all $x \in \Omega$.
In this case, for any $x_{0} \in \Omega$, the kernel score simplifies to

$$
    S_{k}(F, y) = \mathbb{E} \rho(X, y) - \frac{1}{2} \mathbb{E} \rho(X, X^{\prime}),
$$

where $X, X^{\prime} \sim F$ are independent. Note that this holds only up to integrability conditions
that are not relevant for the discussion here, see Sejdinovic et al. (2013).

Kernel scores of this form include:

- The CRPS: $\rho(x, x^{\prime}) = |x - x^{\prime}|$
- The energy score: $\rho(x, x^{\prime}) = \|x - x^{\prime}\|$, where $\| \cdot \|$ is the Euclidean distance on $\mathbb{R}^{d}$
- The variogram score: $\rho(x, x^{\prime}) = \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i, j} \left( |x_{i} - x_{j}|^{p} - |x_{i}^{\prime} - x_{j}^{\prime}|^{p} \right)^{2}$ for some $p > 0$, $h_{i, j} > 0$
- The threshold-weighted CRPS: $\rho(x, x^{\prime}) = |v(x) - v(x^{\prime})|$ for some $v : \mathbb{R} \to \mathbb{R}$
- The threshold-weighted energy score : $\rho(x, x^{\prime}) = \|v(x) - v(x^{\prime})\|$ for some $v : \mathbb{R}^{d} \to \mathbb{R}^{d}$

When the forecast is an ensemble forecast, these kernel scores simplify to

$$
    S_{k}(F_{M}, y) = \frac{1}{M} \sum_{i=1}^{M} \rho(x_{i}, y) - \frac{1}{2 M^{2}} \sum_{i=1}^{M} \sum_{j=1}^{M} \rho(x_{i}, x_{j}).
$$

However, these expressions of kernel scores are generally not fair.
In particular, assuming $x_{1}, \dots, x_{M}$ are independent random draws from $F$,

$$ \mathbb{E} S_{k}(F_{M}, y) = \mathbb{E} S_{k}(F, y) + \frac{1}{2M} \mathbb{E} \rho(x_{1}, x_{2}). $$

This was shown for the CRPS by Ferro et al. (2008), and extends trivially to any kernel score of this form.
However, fair versions of these kernel scores can be obtained by replacing the double expectation
term with an unbiased sample mean:

$$
    S_{k}^{f}(F_{M}, y) = \frac{1}{M} \sum_{i=1}^{M} \rho(x_{i}, y) - \frac{1}{2 M(M - 1)} \sum_{i=1}^{M} \sum_{j=1}^{M} \rho(x_{i}, x_{j}).
$$

Some examples are given below.

More generally, for any positive definite kernel $k$, we have that

$$ \mathbb{E} S_{k}(F_{M}, y) = \mathbb{E} S_{k}(F, y) + \frac{1}{2M} \left( \mathbb{E} k(x_{1}, x_{1}) - \mathbb{E} k(x_{1}, x_{2}) \right). $$

Using this, a fair version of the kernel score is

$$
    S_{k}^{f}(F_{M}, y) = \frac{1}{M(M - 1)} \sum_{i=1}^{M-1} \sum_{j=i+1}^{M} k(x_{i}, x_{j}) + \frac{1}{2} k(y, y) - \frac{1}{M} \sum_{i=1}^{M} k(x_{i}, y).
$$

The first term on the right-hand side is simply the mean of $k(x_i, x_j)$ over all $i,j = 1, \dots, M$ such that $i \neq j$.

For kernels defined in terms of conditionally negative definite functions, as described above, we have that

$$
\mathbb{E} k(x_{1}, x_{1}) - \mathbb{E} k(x_{1}, x_{2}) = 2 \mathbb{E} \rho(x_1, x_{0}) - \mathbb{E} \left[ \rho(x_1, x_0) + \rho(x_2, x_{0}) - \rho(x_1, x_2) \right] = \mathbb{E} \rho(x_1, x_2),
$$

showing that we recover the previous results in this case.


#### CRPS

Plugging $\rho(x, x^{\prime}) = |x - x^{\prime}|$ into the expressions above yields

$$
\mathrm{CRPS}(F_{M}, y) = \frac{1}{M} \sum_{i=1}^{M}|x_i - y| - \frac{1}{2 M^2}\sum_{i=1}^{M} \sum_{j=1}^{M}|x_i - x_j|
$$

(Grimit et al., 2006).

If $x_{1}, \dots, x_{M}$ are instead to be interpreted as a sample from an underlying distribution,
then the forecast can be evaluated with the fair version of the CRPS,

$$
\mathrm{CRPS}^{f}(F_{M}, y) = \frac{1}{M} \sum_{i=1}^{M}|x_i - y| - \frac{1}{2 M (M - 1)}\sum_{i=1}^{M} \sum_{j=1}^{M}|x_i - x_j|
$$

(Ferro et al., 2008). While this expression corresponds to the energy form of the CRPS,
fair representations of the other forms of the CRPS are provided in the following sections.



#### Energy score

The CRPS is one example of a kernel score, and the energy score provides a generalisation of
this to the multivariate case. The energy score is defined as

$$
\mathrm{ES}(F, \boldsymbol{y}) = \mathbb{E} \| \boldsymbol{X} - \boldsymbol{y} \| - \frac{1}{2} \mathbb{E} \| \boldsymbol{X} - \boldsymbol{X}^{\prime} \|,
$$

where $\| \cdot \|$ is the Euclidean distance on $\mathbb{R}^{d}$,
$\boldsymbol{y} \in \Omega \subseteq \mathbb{R}^{d}$, and $\boldsymbol{X}, \boldsymbol{X}^{\prime} \sim F$ are independent,
with $F$ a multivariate predictive distribution on $\Omega$ (Gneiting and Raftery, 2007).

Using the general representations for kernel scores given above, when the forecast is an ensemble
forecast with members $\boldsymbol{x}_{1}, \dots, \boldsymbol{x}_{M} \in \mathbb{R}^{d}$, the energy score becomes

$$
\mathrm{ES}(F_{M}, \boldsymbol{y}) = \frac{1}{M} \sum_{i=1}^{M} \| \boldsymbol{x}_{i} - \boldsymbol{y} \| - \frac{1}{2M^{2}} \sum_{i=1}^{M} \sum_{j=1}^{M} \| \boldsymbol{x}_{i} - \boldsymbol{x}_{j} \|.
$$

The fair version of the energy score is then

$$
\mathrm{ES}^{f}(F_{M}, \boldsymbol{y}) = \frac{1}{M} \sum_{i=1}^{M} \| \boldsymbol{x}_{i} - \boldsymbol{y} \| - \frac{1}{2M(M - 1)} \sum_{i=1}^{M} \sum_{j=1}^{M} \| \boldsymbol{x}_{i} - \boldsymbol{x}_{j} \|.
$$



#### Variogram score

The Variogram score is an alternative multivariate scoring rule that quantifies the difference
between the variogram of the forecast and that of the observation. It is defined as

$$
    \mathrm{VS}_{p}(F, \boldsymbol{y}) = \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left( \mathbb{E} | X_{i} - X_{j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2},
$$

where $p > 0$, $\boldsymbol{y} = (y_{1}, \dots, y_{d}) \in \Omega \subseteq \mathbb{R}^{d}$,
$\boldsymbol{X} = (X_{1}, \dots, X_{d}) \sim F$, with $F$ a multivariate predictive distribution on $\Omega$,
and $h_{i,j} \ge 0$ are weights assigned to different pairs of dimensions (Scheuerer and Hamill, 2015).

Allen et al. (2024) show that the Variogram score is also a kernel score.
In particular, the Variogram score can alternatively be expressed as

$$
    \mathrm{VS}_{p}(F, \boldsymbol{y}) = \mathbb{E} \left[ \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| X_{i} - X_{j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2} \right] - \frac{1}{2} \mathbb{E} \left[ \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| X_{i} - X_{j} |^{p} - | X_{i}^{\prime} - X_{j}^{\prime} |^{p} \right)^{2} \right],
$$

where $\boldsymbol{X} = (X_{1}, \dots, X_{d}), \boldsymbol{X}^{\prime} = (X^{\prime}_{1}, \dots, X^{\prime}_{d}) \sim F$ are independent.

Using, this, when $F$ is an ensemble forecast with members $\boldsymbol{x}_{1}, \dots, \boldsymbol{x}_{M} \in \mathbb{R}^{d}$,
the Variogram score becomes

$$
\begin{align}
    \mathrm{VS}_{p}(F_{M}, \boldsymbol{y}) &= \frac{1}{M} \sum_{m=1}^{M} \left[ \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| x_{m,i} - x_{m,j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2} \right] - \frac{1}{2M^{2}} \sum_{m=1}^{M} \sum_{k=1}^{M} \left[ \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| x_{m,i} - x_{m,j} |^{p} - | x_{k,i} - x_{k,j} |^{p} \right)^{2} \right] \\
    &= \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left( \frac{1}{M}\sum_{m=1}^{M} | x_{m,i} - x_{m,j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2}, \\
\end{align}
$$

which is given by Scheuerer and Hamill (2015), while its fair version is

$$
\begin{align}
    \mathrm{VS}_{p}^{f}(F_{M}, \boldsymbol{y}) &= \frac{1}{M} \sum_{m=1}^{M} \left[ \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| x_{m,i} - x_{m,j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2} \right] - \frac{1}{2M(M - 1)} \sum_{m=1}^{M} \sum_{k=1}^{M} \left[ \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left(| x_{m,i} - x_{m,j} |^{p} - | x_{k,i} - x_{k,j} |^{p} \right)^{2} \right] \\
    &= \sum_{i=1}^{d} \sum_{j=1}^{d} h_{i,j} \left[|y_{i} - y_{j} |^{2p} + \frac{2}{M(M - 1)} \sum_{m=1}^{M - 1} \sum_{k=m+1}^{M} | x_{m,i} - x_{m,j} |^{p} | x_{k,i} - x_{k,j} |^{p} - \frac{2}{M} | y_{i} - y_{j} |^{p} \sum_{m=1}^{M} |x_{m, i} - x_{m, j} |^{p} \right].
\end{align}
$$


#### Threshold-weighted CRPS

The threshold-weighted CRPS is a weighted scoring rule that allows more weight to be assigned
to particular outcomes (Matheson and Winkler, 1976; Gneiting and Ranjan, 2011).
The threshold-weighted CRPS can also be expressed as a kernel score of the above form. In particular,

$$
\begin{align}
\mathrm{twCRPS}(F, y; w) &= \int_{-\infty}^{\infty} (F(x) - 1\{y \le x\})^{2} w(x) dx \\
&= \mathbb{E} | v(X) - v(y) | - \frac{1}{2} \mathbb{E} |v(X) - v(X^{\prime}) |,
\end{align}
$$

where $y \in \Omega \subseteq \mathbb{R}$,  $X, X^{\prime} \sim F$ are independent,
$w : \mathbb{R} \to [0, \infty)$ is a non-negative weight function, and
$v : \mathbb{R} \to \mathbb{R}$ is an anti-derivative of the weight function $w$.
That is, $v(x) - v(x^{\prime}) = \int_{x^{\prime}}^{x} w(z) dz$ for all $x, x^{\prime} \in \mathbb{R}$.

Using the second expression above, the threshold-weighted CRPS for an ensemble forecast
with members $x_{1}, \dots, x_{M} \in \mathbb{R}$ is

$$
\mathrm{twCRPS}(F_{M}, y; w) = \frac{1}{M} \sum_{i=1}^{M} |v(x_{i}) - v(y)| - \frac{1}{2M^{2}} \sum_{i=1}^{M} \sum_{j=1}^{M} |v(x_{i}) - v(x_{j})|
$$

(Allen et al., 2024), and the fair version of the threshold-weighted CRPS becomes

$$
\mathrm{twCRPS}^{f}(F_{M}, y; w) = \frac{1}{M} \sum_{i=1}^{M} |v(x_{i}) - v(y)| - \frac{1}{2M(M - 1)} \sum_{i=1}^{M} \sum_{j=1}^{M} |v(x_{i}) - v(x_{j})|.
$$


#### Threshold-weighted energy score

Similarly, the threshold-weighted energy score is a kernel score, defined as

$$
\mathrm{twES}(F, \boldsymbol{y}; v) = \mathbb{E} \| v(\boldsymbol{X}) - v(\boldsymbol{y}) \| - \frac{1}{2} \mathbb{E} \|v(\boldsymbol{X}) - v(\boldsymbol{X}^{\prime}) \|,
$$
where $\boldsymbol{y} \in \Omega \subseteq \mathbb{R}^{d}$, $\boldsymbol{X}, \boldsymbol{X}^{\prime} \sim F$ are independent, and $v : \Omega \to \Omega$ (Allen et al., 2024).

The threshold-weighted energy score for an ensemble forecast with members $\boldsymbol{x}_{1}, \dots, \boldsymbol{x}_{M} \in \mathbb{R}^{d}$ is then

$$
\mathrm{twES}(F_{M}, \boldsymbol{y}; v) = \frac{1}{M} \sum_{i=1}^{M} \|v(\boldsymbol{x}_{i}) - v(\boldsymbol{y})\| - \frac{1}{2M^{2}} \sum_{i=1}^{M} \sum_{j=1}^{M} \|v(\boldsymbol{x}_{i}) - v(\boldsymbol{x}_{j})\|.
$$

The fair version of the threshold-weighted energy score becomes

$$
\mathrm{twES}^{f}(F_{M}, \boldsymbol{y}; v) = \frac{1}{M} \sum_{i=1}^{M} \|v(\boldsymbol{x}_{i}) - v(\boldsymbol{y})\| - \frac{1}{2M(M - 1)} \sum_{i=1}^{M} \sum_{j=1}^{M} \|v(\boldsymbol{x}_{i}) - v(\boldsymbol{x}_{j})\|.
$$


### Fair versions of CRPS forms

The various representations of the CRPS also lead to multiple ways of calculating the fair CRPS.
These representations all yield the same score, but again differ in their computational complexity;
the fair forms all have the same complexity as the original forms given previously.


#### Energy form (NRG)

The energy form of the fair CRPS, given already above, is

$$
\mathrm{CRPS}_{\text{NRG}}^{f}(F_{M}, y) = \frac{1}{M} \sum_{i=1}^{M}|x_i - y| - \frac{1}{2 M(M - 1)}\sum_{i=1}^{M} \sum_{j=1}^{M}|x_i - x_j|.
$$


#### Quantile decomposition form (QD)

A quantile decomposition form of the fair CRPS is

$$
\mathrm{CRPS}_{\mathrm{QD}}^{f}(F_{M}, y) = \frac{2}{M} \sum_{i=1}^{M}\left[\mathbb{1}\{y \le x_{(i)}\} - \frac{i - 1}{M - 1} \right] (x_{(i)} - y).
$$


#### Probability weighted moment form (PWM)

A probability weighted moment form of the fair CRPS is

$$
\mathrm{CRPS}_{\mathrm{PWM}}^{f}(F_{M}, y) = \frac{1}{M} \sum_{i=1}^{M}|x_{(i)} - y| + \hat{\beta_0} - 2\hat{\beta_1},
$$

where $\hat{\beta_0} = \frac{1}{M} \sum_{i=1}^{M}x_{(i)}$ and $\hat{\beta_1} = \frac{1}{M(M-1)} \sum_{i=1}^{M}(i - 1)x_{(i)}$.
This was proposed by Taillardat et al. (2016).


#### Integral form (INT)

The integral form of the CRPS can also be adapted to recover the fair version,

$$
\mathrm{CRPS}_{\text{INT}}^{f}(F_{M}, y) = \int_{\mathbb{R}} \left[ \mathbb{1}\{y \le x\} +
\frac{2}{M(M - 1)} \sum_{i=1}^{M - 1} \sum_{j=i+1}^{M} \mathbb{1}\{x_i \le x \}\mathbb{1}\{x_j \le x \} -
\frac{2}{M} \mathbb{1}\{y \le x\} \sum_{i=1}^{M} \mathbb{1}\{x_i \le x \} \right] dx.
$$

This corresponds to the original CRPS with the integrand expanded out and with the components
in the double summation removed when $i = j$. This also requires numerical approximation of
the integral, so can again introduce small errors to the score.
