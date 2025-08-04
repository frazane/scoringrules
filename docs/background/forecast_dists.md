(forecast-dists)=
# Parametric forecast distributions

Probabilistic forecasts take the form of probability distributions over the set of possible outcomes.
When the outcome is real-valued ($Y \in \mathbb{R}$), probabilistic forecasts often come in the
form of parametric distributions. The `scoringrules` package provides the functionality to calculate
the CRPS and Log score for a range of parametric distributions. In this document, we list the
distributions that are available, along with their probability density/mass function (pdf/pmf),
cumulative distribution function (cdf), and an analytical formula for their CRPS. The Log score
is simply the negative logarithm of the pdf/pmf, when this exists. The analytical
formulae for the CRPS are all taken from Appendix A of the scoringRules R package (Jordan et al., 2017).


## Discrete outcomes

### Binomial distribution

The *binomial distribution* has two parameters that describe the number of trials, $n = 0, 1, \dots, $
and the probability of a success, $p \in [0, 1]$. Its pmf, cdf, and CRPS are

$$
f_{n, p}(x) =
\begin{cases}
    {n \choose x} p^{x} (1 - p)^{n - x}, & x = 0, 1, \dots, n, \\
    0, & \text{otherwise,}
\end{cases}
$$

$$
F_{n, p}(x) =
\begin{cases}
    I(n - \lfloor x \rfloor, \lfloor x \rfloor + 1, 1 - p), & x \ge 0, \\
    0, & x < 0,
\end{cases}
$$

$$
\mathrm{CRPS}(F_{n, p}, y) = 2 \sum_{x = 0}^{n} f_{n,p}(x) (1\{y < x\}
 - F_{n,p}(x) + f_{n,p}(x)/2) (x - y),
$$

where $I(a, b, x)$ is the regularised incomplete beta function.


### Hypergeometric distribution

The *hypergeometric distribution* has three parameters that describe the number of objects with
the relevant feature, $m = 0, 1, \dots, $ the number of objects without this feature, $n = 0, 1, \dots $,
and the size of the sample to be drawn without replacement from these objects, $k = 0, 1, \dots, m + n$.
Its pmf, cdf, and CRPS are

$$
f_{m, n, k}(x) =
\begin{cases}
    \frac{{m \choose x} {n \choose k-x}}{m + n \choose k}, & x = \max{0, k - n}, \dots, \min{k, m}, \\
    0, & \text{otherwise,}
\end{cases}
$$

$$
F_{m, n, k}(x) =
\begin{cases}
    \sum_{i = 0}^{\lfloor x \rfloor} f_{m, n, k}(i), & x \ge 0, \\
    0, & x < 0,
\end{cases}
$$

$$
\mathrm{CRPS}(F_{m, n, k}, y) = 2 \sum_{x = 0}^{n} f_{m,n,k}(x) (1\{y < x\}
        - F_{m,n,k}(x) + f_{m,n,k}(x)/2) (x - y).
$$


### Negative binomial distribution

The *negative binomial distribution* has two parameters that describe the number of successes,
$n > 0$, and the probability of a success, $p \in [0, 1]$. Its pmf, cdf, and CRPS are

$$
f_{n, p}(x) =
\begin{cases}
    \frac{\Gamma(x + n)}{\Gamma(n) x!} p^{n} (1 - p)^{x} , & x = 0, 1, 2, \dots, \\
    0, & \text{otherwise,}
\end{cases}
$$

$$
F_{n, p}(x) =
\begin{cases}
    I(n, \lfloor x + 1 \rfloor, p), & x \ge 0, \\
    0, & x < 0,
\end{cases}
$$

$$
\mathrm{CRPS}(F_{n, p}, y) = y (2 F_{n, p}(y) - 1) - \frac{n(1 - p)}{p^{2}} \left( p (2 F_{n+1, p}(y - 1) - 1) +
    _{2} F_{1} \left( n + 1, \frac{1}{2}; 2; -\frac{4(1 - p)}{p^{2}} \right) \right),
$$

where $I(a, b, x)$ is the regularised incomplete beta function, and $_2 F_{1}(a, b; c; x)$ is the hypergeometric function.


### Poisson distribution

The *Poisson distribution* has one mean parameter, $\lambda > 0$. Its pmf, cdf, and CRPS are

$$
f_{\lambda}(x) =
\begin{cases}
    \frac{\lambda^{x}}{x!}e^{-\lambda}, & x = 0, 1, 2, \dots, \\
    0, & \text{otherwise,}
\end{cases}
$$

$$
F_{\lambda}(x) =
\begin{cases}
    \frac{\Gamma_{u}(\lfloor x + 1 \rfloor, \lambda)}{\Gamma(\lfloor x + 1 \rfloor)}, & x \ge 0, \\
    0, & x < 0,
\end{cases}
$$

$$
\mathrm{CRPS}(F_{\lambda}, y) = (y - \lambda) (2F_{\lambda}(y) - 1)
        + 2 \lambda f_{\lambda}(\lfloor y \rfloor )
        - \lambda e^{-2 \lambda} (I_{0} (2 \lambda) + I_{1} (2 \lambda)),
$$

where $\Gamma_{u}(a, x)$ is the upper incomplete gamma function, and $I_{m}(x)$ is the
modified Bessel function of the first kind.



## Continuous real-valued outcomes

### Laplace distribution

The *Laplace distribution* has one location parameter, $\mu \in \mathbb{R}$, and one scale parameter $\sigma > 0$.
Its pdf, cdf, and CRPS are

$$
f_{\mu, \sigma}(x) = \frac{1}{2\sigma} \exp \left( - \frac{|x - \mu|}{\sigma} \right)
$$

$$
F_{\mu, \sigma}(x) =
\begin{cases}
    \frac{1}{2} \exp \left( \frac{x - \mu}{\sigma} \right), & x \le \mu, \\
    1 - \frac{1}{2} \exp \left( - \frac{x - \mu}{\sigma} \right), & x \ge \mu, \\
\end{cases}
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1}, y) &= |y| + \exp(-|y|) - \frac{3}{4}, \\
    \mathrm{CRPS}(F_{\mu, \sigma}, y) &= \sigma \mathrm{CRPS} \left( F_{0, 1}, \frac{y - \mu}{\sigma} \right).
\end{align}
$$


### Logistic distribution

The *Logistic distribution* has one location parameter, $\mu \in \mathbb{R}$, and one scale parameter, $\sigma > 0$.
Its pdf, cdf, and CRPS are

$$
f_{\mu, \sigma}(x) = \frac{\exp \left( - \frac{x - \mu}{\sigma} \right)}{\sigma \left( 1 + \exp \left( - \frac{x - \mu}{\sigma} \right) \right)^{2}},
$$

$$
F_{\mu, \sigma}(x) =  \frac{1}{1 + \exp \left( - \frac{x - \mu}{\sigma} \right)},
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1}, y) &= y - 2 \log \left( F_{0, 1}(y) \right) - 1, \\
    \mathrm{CRPS}(F_{\mu, \sigma}, y) &= \sigma \mathrm{CRPS} \left( F_{0, 1}, \frac{y - \mu}{\sigma} \right).
\end{align}
$$


### Normal distribution

The *normal distribution* has one location parameter, $\mu \in \mathbb{R}$, and one scale parameter, $\sigma > 0$.
Its pdf, cdf, and CRPS are

$$
f_{\mu, \sigma}(x) = \frac{1}{\sigma} \phi \left( \frac{x - \mu}{\sigma} \right) = \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( - \frac{(x - \mu)^{2}}{2\sigma^{2}} \right),
$$

$$
F_{\mu, \sigma}(x) = \Phi \left( \frac{x - \mu}{\sigma} \right) = \frac{1}{2} \left[ 1 + \text{erf} \left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right],
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1}, y) &= y (2\Phi(y) - 1) + 2 \phi(y) - \frac{1}{\sqrt{\pi}}, \\
    \mathrm{CRPS}(F_{\mu, \sigma}, y) &= \sigma \mathrm{CRPS} \left( F_{0, 1}, \frac{y - \mu}{\sigma} \right),
\end{align}
$$

where $\phi$ and $\Phi$ represent the standard normal pdf and cdf, respectively, and
$\text{erf}$ is the error function.


### Mixture of normal distributions

A *mixture of normal distributions* has $M$ location parameters, $\mu_{1}, \dots, \mu_{M} \in \mathbb{R}$,
$M$ scale parameters, $\sigma_{1}, \dots, \sigma_{M} > 0$, and $M$ weight parameters, $w_{1}, \dots, w_{M} \ge 0$
that sum to one. Its pdf, cdf, and CRPS are

$$
f(x) = \sum_{m=1}^{M} \frac{w_{m}}{\sigma_{m}} \phi \left( \frac{x - \mu_{m}}{\sigma_{m}} \right),
$$

$$
F(x) = \sum_{m=1}^{M} w_{m} \Phi \left( \frac{x - \mu_{m}}{\sigma_{m}} \right),
$$

$$
\mathrm{CRPS}(F, y) = \sum_{m=1}^{M} w_{m} A(y - \mu_{m}, \sigma_{m}^{2}) - \frac{1}{2} \sum_{m=1}^{M} \sum_{k=1}^{M} w_{m} w_{k} A(\mu_{m} - \mu_{k}, \sigma_{m}^{2} + \sigma_{k}^{2}),
$$

where $A(\mu, \sigma^{2}) = \mu (2 \Phi(\frac{\mu}{\sigma}) - 1) + 2\sigma \phi(\frac{\mu}{\sigma})$,
with $\phi$ and $\Phi$ the standard normal pdf and cdf, respectively.


### Student's t distribution

The (generalised) *Student's t distribution* has one degrees of freedom parameter, $\nu > 0$,
one location parameter, $\mu \in \mathbb{R}$, and one scale parameter, $\sigma > 0$.
Its pdf, cdf, and CRPS are

$$
f_{\nu, \mu, \sigma}(x) = \frac{1}{\sigma \sqrt{\nu} B(\frac{1}{2}, \frac{\nu}{2})} \left( 1 + \frac{(x - \mu)^{2}}{\sigma^{2} \nu} \right)^{- \frac{\nu + 1}{2}},
$$

$$
F_{\nu, \mu, \sigma}(x) = \frac{1}{2} + \left( \frac{x - \mu}{\sigma} \right) \frac{_{2} F_{1} (\frac{1}{2}, \frac{\nu + 1}{2} ; \frac{3}{2} ; - \frac{(x - \mu)^{2}}{\sigma^{2} \nu})}{ \sqrt{\nu} B(\frac{1}{2}, \frac{\nu}{2}) },
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{\nu, 0, 1}, y) &= y (2F_{\nu, 0, 1}(y) - 1) + 2 f_{\nu, 0, 1}(y) \left( \frac{\nu + y^{2}}{\nu - 1} \right) - \frac{2 \sqrt{\nu}}{\nu - 1} \frac{B(\frac{1}{2}, \nu - \frac{1}{2})}{B(\frac{1}{2}, \frac{\nu}{2})^{2}}, \\
    \mathrm{CRPS}(F_{\nu, \mu, \sigma}, y) &= \sigma \mathrm{CRPS} \left( F_{\nu, 0, 1}, \frac{y - \mu}{\sigma} \right).
\end{align}
$$


### Two-piece exponential distribution

The *two-piece exponential distribution* has one location parameter, $\mu \in \mathbb{R}$, and
two scale parameters, $\sigma_{1}, \sigma_{2} > 0$. Its pdf, cdf, and CRPS are

$$
f_{\mu, \sigma_{1}, \sigma_{2}}(x) =
\begin{cases}
\frac{1}{\sigma_{1} + \sigma_{2}} \exp \left( - \frac{\mu - x}{\sigma_{1}} \right), & x \le \mu, \\
\frac{1}{\sigma_{1} + \sigma_{2}} \exp \left( - \frac{x - \mu}{\sigma_{2}} \right), & x \ge \mu, \\
\end{cases}
$$

$$
F_{\mu, \sigma_{1}, \sigma_{2}}(x) =
\begin{cases}
    \frac{\sigma_{1}}{\sigma_{1} + \sigma_{2}} \exp \left( - \frac{\mu - x}{\sigma_{1}} \right), & x \le \mu, \\
    1 - \frac{\sigma_{2}}{\sigma_{1} + \sigma_{2}} \exp \left( - \frac{x - \mu}{\sigma_{2}} \right), & x \ge \mu, \\
\end{cases}
$$

$$
\mathrm{CRPS}(F_{\mu, \sigma_{1}, \sigma_{2}}, y) =
\begin{cases}
    |y - \mu| + \frac{2 \sigma_{1}^{2}}{\sigma_{1} + \sigma_{2}} \left[ \exp \left( - \frac{|y - \mu|}{\sigma_{1}} \right) - 1 \right] + \frac{\sigma_{1}^{3} + \sigma_{2}^{3}}{2(\sigma_{1} + \sigma_{2})^{2}}, & y \le \mu, \\
    |y - \mu| + \frac{2 \sigma_{2}^{2}}{\sigma_{1} + \sigma_{2}} \left[ \exp \left( - \frac{|y - \mu|}{\sigma_{2}} \right) - 1 \right] + \frac{\sigma_{1}^{3} + \sigma_{2}^{3}}{2(\sigma_{1} + \sigma_{2})^{2}}, & y \ge \mu.
\end{cases}
$$


### Two-piece normal distribution

The *two-piece normal distribution* has one location parameter, $\mu \in \mathbb{R}$, and
two scale parameters, $\sigma_{1}, \sigma_{2} > 0$. Its pdf, cdf, and CRPS are

$$
f_{\mu, \sigma_{1}, \sigma_{2}}(x) =
\begin{cases}
\frac{2}{\sigma_{1} + \sigma_{2}} \phi \left( \frac{x - \mu}{\sigma_{1}} \right), & x \le \mu, \\
\frac{2}{\sigma_{1} + \sigma_{2}} \phi \left( \frac{x - \mu}{\sigma_{2}} \right), & x \ge \mu, \\
\end{cases}
$$

$$
F_{\mu, \sigma_{1}, \sigma_{2}}(x) =
\begin{cases}
    \frac{2\sigma_{1}}{\sigma_{1} + \sigma_{2}} \Phi \left( \frac{x - \mu}{\sigma_{1}} \right), & x \le \mu, \\
    \frac{\sigma_{1} - \sigma_{2}}{\sigma_{1} + \sigma_{2}} + \frac{2\sigma_{2}}{\sigma_{1} + \sigma_{2}} \Phi \left( \frac{x - \mu}{\sigma_{2}} \right), & x \ge \mu, \\
\end{cases}
$$

$$
\begin{align}
\mathrm{CRPS}(F_{\mu, \sigma_{1}, \sigma_{2}}, y) = & \sigma_{1} \mathrm{CRPS} \left( F_{-\infty, 0}^{0, \sigma_{2}/(\sigma_{1} + \sigma_{2})}, \min \left(0, \frac{y - \mu}{\sigma_{1}} \right) \right) \\
    & + \sigma_{2} \mathrm{CRPS} \left( F^{\infty, 0}_{0, \sigma_{1}/(\sigma_{1} + \sigma_{2})}, \max \left(0, \frac{y - \mu}{\sigma_{2}} \right) \right),
\end{align}
$$

where $\phi$ and $\Phi$ the standard normal pdf and cdf, respectively, and $F_{l, L}^{u, U}$
is the cdf of the truncated and censored normal distribution.


## Continuous positive outcomes

### Exponential distribution

The *exponential distribution* has one rate parameter, $\lambda > 0$. Its pdf, cdf, and CRPS are

$$
f_{\lambda}(x) =
\begin{cases}
    \lambda e^{-\lambda x}, & x \ge 0, \\
    0, & x < 0, \\
\end{cases}
$$

$$
F_{\lambda}(x) =
\begin{cases}
    1 - e^{-\lambda x}, & x \ge 0, \\
    0, & x < 0, \\
\end{cases}
$$

$$
\mathrm{CRPS}(F_{\lambda}, y) = |y| - \frac{2 F_{\lambda}(y)}{\lambda} + \frac{1}{2\lambda}.
$$


### Gamma distribution

The *gamma distribution* has one shape parameter, $\alpha > 0$, and one rate parameter, $\beta > 0$. Its pdf, cdf, and CRPS are

$$
f_{\alpha, \beta}(x) =
\begin{cases}
    \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, & x \ge 0, \\
    0, & x < 0, \\
\end{cases}
$$

$$
F_{\alpha, \beta}(x) =
\begin{cases}
    \frac{\Gamma_{l}(\alpha, \beta x)}{\Gamma(\alpha)} , & x \ge 0, \\
    0, & x < 0, \\
\end{cases}
$$

$$
\mathrm{CRPS}(F_{\alpha, \beta}, y) = y (2 F_{\alpha, \beta}(y) - 1) - \frac{\alpha}{\beta} (2 F_{\alpha + 1, \beta}(y) - 1) - \frac{1}{\beta B(\frac{1}{2}, \alpha)},
$$

where $\Gamma_{l}$ is the lower incomplete gamma function, and $B$ is the beta function.


### Log-Laplace distribution

The *log-Laplace distribution* has one location parameter, $\mu \in \mathbb{R}$, and one scale parameter, $\sigma > 0$.
Its pdf, cdf, and CRPS are

$$
f_{\mu, \sigma}(x) =
\begin{cases}
    \frac{1}{2 \sigma x} \exp \left( - \frac{| \log x - \mu |}{\sigma} \right), & x > 0, \\
    0, & x \le 0, \\
\end{cases}
$$

$$
F_{\mu, \sigma}(x) =
\begin{cases}
    0, & x \le 0, \\
    \frac{1}{2} \exp \left( \frac{\log x - \mu}{\sigma} \right) , & 0 < \log x < \mu, \\
    1 - \frac{1}{2} \exp \left( - \frac{\log x - \mu}{\sigma} \right), & \log x \ge \mu, \\
\end{cases}
$$

$$
\mathrm{CRPS}(F_{\mu, \sigma}, y) = y (2 F_{\mu, \sigma}(y) - 1) + e^{\mu} \left( \frac{\sigma}{4 - \sigma^{2}} + A(y) \right),
$$

where

$$
A(x) =
\begin{cases}
    \frac{1}{1 + \sigma} \left( 1 - (2 F_{\mu, \sigma}(x))^{1 + \sigma} \right), & \log x < \mu, \\
    -\frac{1}{1 - \sigma} \left( 1 - (2 (1 - F_{\mu, \sigma}(x)))^{1 - \sigma} \right), & \log x \ge \mu. \\
\end{cases}
$$


### Log-logistic distribution

The *log-logistic distribution* has one location parameter, $\mu \in \mathbb{R}$, and one scale parameter, $\sigma > 0$.
Its pdf, cdf, and CRPS are

$$
f_{\mu, \sigma}(x) =
\begin{cases}
    \frac{\exp \left(\frac{\log x - \mu}{\sigma} \right)}{\sigma x \left(1 + \exp \left(\frac{\log x - \mu}{\sigma} \right) \right)^{2}}, & x > 0, \\
    0, & x \le 0, \\
\end{cases}
$$

$$
F_{\mu, \sigma}(x) =
\begin{cases}
    \left(1 + \exp \left( - \frac{\log x - \mu}{\sigma} \right) \right)^{-1}, & x > 0, \\
    0, & x \le 0, \\
\end{cases}
$$

$$
\mathrm{CRPS}(F_{\mu, \sigma}, y) = y (2 F_{\mu, \sigma}(y) - 1) - e^{\mu} B(1 + \sigma, 1 - \sigma) ( 2I(1 + \sigma, 1 - \sigma, F_{\mu, \sigma}(y)) + \sigma - 1 ),
$$

where $B$ is the beta function, and $I$ is the regularised incomplete beta function.


### Log-normal distribution

The *log-normal distribution* has one location parameter, $\mu \in \mathbb{R}$, and one scale parameter, $\sigma > 0$.
Its pdf, cdf, and CRPS are

$$
f_{\mu, \sigma}(x) =
\begin{cases}
    \frac{1}{\sigma x} \phi \left( \frac{\log x - \mu}{\sigma} \right) = \frac{1}{\sigma x \sqrt{2 \pi}} \exp \left( - \frac{(\log x - \mu)^{2}}{2\sigma^{2}} \right), & x > 0, \\
    0, & x \le 0, \\
\end{cases}
$$

$$
F_{\mu, \sigma}(x) =
\begin{cases}
    \Phi \left( \frac{\log x - \mu}{\sigma} \right) = \frac{1}{2} \left[ 1 + \text{erf} \left( \frac{\log x - \mu}{\sigma \sqrt{2}} \right) \right], & x > 0, \\
    0, & x \le 0, \\
\end{cases}
$$

$$
\mathrm{CRPS}(F_{\mu, \sigma}, y) = y (2 F_{\mu, \sigma}(y) - 1) - 2 \exp \left( \mu + \frac{\sigma^{2}}{2} \right) \left( \Phi \left( \frac{\log y - \mu - \sigma^{2}}{\sigma} \right) + \Phi \left( \frac{\sigma}{\sqrt{2}} \right) - 1 \right),
$$

where $\phi$ and $\Phi$ represent the standard normal pdf and cdf, respectively, and $\text{erf}$ is the error function.


## Bounded or censored outcomes

### Beta distribution

The (generalised) *beta distribution* has two shape parameters, $\alpha, \beta > 0$, and
lower and upper bound parameters $l,u \in \mathbb{R}$, such that $l < u$. Its pdf, cdf, and CRPS are

$$
f_{\alpha, \beta}^{l, u}(x) =
\begin{cases}
    \frac{1}{B(\alpha, \beta)} \left( \frac{x - l}{u - l} \right)^{\alpha - 1} \left(\frac{u - x}{u - l} \right)^{\beta - 1} , & l \le x \le u, \\
    0, & x < l \quad \text{or} \quad x > u, \\
\end{cases}
$$

$$
F_{\alpha, \beta}^{l, u}(x) =
\begin{cases}
    0, & x < l, \\
    I \left( \alpha, \beta, \frac{x - l}{u - l} \right), & l \le x \le u, \\
    1, & x > u, \\
\end{cases}
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{\alpha, \beta}^{0, 1}, y) &= y (2F_{\alpha, \beta}^{0, 1}(y) - 1) + \frac{\alpha}{\alpha + \beta} \left( 1 - 2 F_{\alpha + 1, \beta}^{0, 1}(y) - \frac{2B(2 \alpha, 2 \beta)}{\alpha B(\alpha, \beta)^{2}} \right), \\
    \mathrm{CRPS}(F_{\alpha, \beta}^{l, u}, y) &= (u - l) \mathrm{CRPS} \left( F_{\alpha, \beta}^{0, 1}, \frac{y - l}{u - l} \right),
\end{align}
$$

where $B$ is the beta function, and $I$ is the regularised incomplete beta function.


### Continuous uniform distribution

The continuous *uniform distribution* has lower and upper bound parameters $l,u \in \mathbb{R}$, $l < u$,
and point mass parameters $L, U \ge 0$, $L + U < 1$ that assign mass to the boundary points.
Its pdf+pmf, cdf, and CRPS are

$$
f_{l, u}^{L, U}(x) =
\begin{cases}
    0, & x < l \quad \text{or} \quad x > u, \\
    L, & x = l, \\
    U, & x = u, \\
    \frac{1 - L - U}{u - l}, & l < x < u, \\
\end{cases}
$$

$$
F_{l, u}^{L, U}(x) =
\begin{cases}
    0, & x < l, \\
    L + (1 - L - U) \frac{x - l}{u - l}, & l \le x \le u, \\
    1, & x > u, \\
\end{cases}
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1}^{L, U}, y) &= | y - F(y) | + F(y)^{2}(1 - L - U) - F(y)(1 - 2L) + \frac{(1 - L - U)^{2}}{3} + (1 - L)U, \\
    \mathrm{CRPS}(F_{l, u}^{L, U}, y) &= (u - l) \mathrm{CRPS} \left( F_{0, 1}^{L, U}, \frac{y - l}{u - l} \right),
\end{align}
$$

where $F = F_{0, 1}^{0, 0}$ is the standard uniform distribution function.


### Exponential distribution with point mass

The exponential distribution can be generalised to different supports, and to include a
point mass at the boundary. This *generalised exponential distribution* has one location parameter, $\mu \in \mathbb{R}$,
one scale parameter $\sigma > 0$, and one point mass parameter $M \in [0, 1]$. Its pdf+pmf, cdf, and CRPS are

$$
f_{\mu, \sigma, M}(x) =
\begin{cases}
    0, & x < \mu, \\
    M, & x = \mu, \\
    \frac{1 - M}{\sigma} \exp \left( - \frac{x - \mu}{\sigma} \right), & x > \mu, \\
\end{cases}
$$

$$
F_{\mu, \sigma, M}(x) =
\begin{cases}
    0, & x < \mu, \\
    M + (1 - M) \left(1 - \exp \left( - \frac{x - \mu}{\sigma} \right) \right), & x \ge \mu, \\
\end{cases}
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1, M}, y) &= |y| - 2(1 - M)F(y) + \frac{(1 - M)^{2}}{2}, \\
    \mathrm{CRPS}(F_{\mu, \sigma, M}, y) &= \sigma \mathrm{CRPS} \left( F_{0, 1, M}, \frac{y - \mu}{\sigma} \right),
\end{align}
$$

where $F = F_{0, 1, 0}$ is the standard exponential distribution function.


### Generalised extreme value (GEV) distribution

The *generalised extreme value (GEV) distribution* has one location parameter, $\mu \in \mathbb{R}$,
one scale parameter, $\sigma > 0$, and one shape parameter, $\xi \in \mathbb{R}$. The GEV distribution can be divided
into three types, depending on the value of $\xi$. The pdf, cdf, and CRPS for each type are:

If $\xi = 0$,

$$
f_{\mu, \sigma, \xi}(x) = \frac{1}{\sigma} \exp \left( - \frac{x - \mu}{\sigma} \right) \exp \left( - \exp \left( - \frac{x - \mu}{\sigma} \right) \right),
$$

$$
F_{\mu, \sigma, \xi}(x) = \exp \left( - \exp \left( - \frac{x - \mu}{\sigma} \right) \right),
$$

$$
\mathrm{CRPS}(F_{0, 1, \xi}, y) = - y - 2 \text{Ei}(\log F_{0, 1, \xi}(y)) + \gamma - \log 2,
$$

where $\text{Ei}$ is the exponential integral, and $\gamma$ is the Euler-Mascheroni constant.

If $\xi > 0$,

$$
f_{\mu, \sigma, \xi}(x) =
\begin{cases}
    0, & x < \mu - \frac{\sigma}{\xi}, \\
    \frac{1}{\sigma} \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{-(\xi + 1)/\xi} \exp \left( - \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{-1/\xi} \right), & x \ge \mu - \frac{\sigma}{\xi},
\end{cases}
$$

$$
F_{\mu, \sigma, \xi}(x) =
\begin{cases}
    0, & x < \mu - \frac{\sigma}{\xi}, \\
    \exp \left( - \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{-1/\xi} \right), & x \ge \mu - \frac{\sigma}{\xi},
\end{cases}
$$

$$
\mathrm{CRPS}(F_{0, 1, \xi}, y) = y (2 F_{0, 1, \xi}(y) - 1) - 2 G_{\xi} (y) - \frac{1 - (2 - 2^{\xi}) \Gamma(1 - \xi)}{\xi},
$$

where

$$
G_{\xi}(x) =
\begin{cases}
    0, & x \le -\frac{1}{\xi}, \\
    -\frac{F_{0, 1, \xi}(x)}{\xi} + \frac{\Gamma_{u}(1 - \xi, - \log F_{0, 1, \xi}(x))}{\xi}, & x > -\frac{1}{\xi},
\end{cases}
$$

with $\Gamma$ the gamma function, and $\Gamma_{u}$ the upper incomplete gamma function.


If $\xi < 0$,

$$
f_{\mu, \sigma, \xi}(x) =
\begin{cases}
    \frac{1}{\sigma} \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{-(\xi + 1)/\xi} \exp \left( - \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{-1/\xi} \right), & x \le \mu - \frac{\sigma}{\xi}, \\
    0, & x > \mu - \frac{\sigma}{\xi},
\end{cases}
$$

$$
F_{\mu, \sigma, \xi}(x) =
\begin{cases}
    \exp \left( - \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{-1/\xi} \right), & x \le \mu - \frac{\sigma}{\xi}, \\
    1, & x > \mu - \frac{\sigma}{\xi},
\end{cases}
$$

$$
\mathrm{CRPS}(F_{0, 1, \xi}, y) = y (2 F_{0, 1, \xi}(y) - 1) - 2 G_{\xi} (y) - \frac{1 - (2 - 2^{\xi}) \Gamma(1 - \xi)}{\xi},
$$

where

$$
G_{\xi}(x) =
\begin{cases}
    -\frac{F_{0, 1, \xi}(x)}{\xi} + \frac{\Gamma_{u}(1 - \xi, - \log F_{0, 1, \xi}(x))}{\xi}, & x < -\frac{1}{\xi}, \\
    -\frac{1}{\xi} + \frac{\Gamma(1 - \xi)}{\xi}, & x \ge -\frac{1}{\xi},
\end{cases}
$$

with $\Gamma$ the gamma function, and $\Gamma_{u}$ the upper incomplete gamma function.

For all $\xi$, we have that

$$
\mathrm{CRPS}(F_{\mu, \sigma, \xi}, y) = \sigma \mathrm{CRPS} \left( F_{0, 1, \xi}, \frac{y - \mu}{\sigma} \right).
$$


### Generalised Pareto distribution (GPD) with point mass

The *generalised Pareto distribution (GPD)* has one location parameter, $\mu \in \mathbb{R}$,
one scale parameter, $\sigma > 0$, one shape parameter, $\xi \in \mathbb{R}$, and one point mass
parameter at the lower boundary, $M \in [0, 1]$. Its pdf+pmf, cdf, and CRPS are:

If $\xi = 0$,

$$
f_{\mu, \sigma, \xi, M}(x) =
\begin{cases}
    0, & x < \mu, \\
    M, & x = \mu, \\
    \frac{1 - M}{\sigma} \exp \left( - \frac{x - \mu}{\sigma} \right), & x > \mu,
\end{cases}
$$

$$
F_{\mu, \sigma, \xi, M}(x) =
\begin{cases}
    0, & x < \mu, \\
    M + (1 - M) \left( 1 - \exp \left( - \frac{x - \mu}{\sigma} \right) \right), & x \ge \mu.
\end{cases}
$$

If $\xi > 0$,

$$
f_{\mu, \sigma, \xi, M}(x) =
\begin{cases}
    0, & x < \mu, \\
    M, & x = \mu, \\
    \frac{1 - M}{\sigma} \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{- (\xi + 1) / \xi}, & x > \mu,
\end{cases}
$$

$$
F_{\mu, \sigma, \xi, M}(x) =
\begin{cases}
    0, & x < \mu, \\
    M + (1 - M) \left( 1 - \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{- 1 / \xi} \right), & x \ge \mu.
\end{cases}
$$

If $\xi < 0$,

$$
f_{\mu, \sigma, \xi, M}(x) =
\begin{cases}
    M, & x = \mu, \\
    \frac{1 - M}{\sigma} \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{- (\xi + 1) / \xi}, & \mu < x \le \mu - \frac{\sigma}{\xi}, \\
    0, & \text{otherwise},
\end{cases}
$$

$$
F_{\mu, \sigma, \xi, M}(x) =
\begin{cases}
    0, & x < \mu, \\
    M + (1 - M) \left( 1 - \left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{- 1 / \xi} \right), & \mu < x \le \mu - \frac{\sigma}{\xi}, \\
    1, & x > \mu - \frac{\sigma}{\xi}.
\end{cases}
$$

For all $\xi$,

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1, \xi, M}, y) &= |y| - \frac{2(1 - M)}{1 - \xi} \left( 1 - (1 - F_{0, 1, \xi, 0}(y))^{1 - \xi} \right) + \frac{(1 - M)^{2}}{2 - \xi}, \\
    \mathrm{CRPS}(F_{\mu, \sigma, \xi, M}, y) &= \sigma \mathrm{CRPS} \left( F_{0, 1, \xi, M}, \frac{y - \mu}{\sigma} \right).
\end{align}
$$


### Generalised truncated or censored logistic distribution

The logistic distribution can be generalised to account for truncation, censoring, and to
allow point masses at the boundary points of its support. This *generalised logistic distribution*
has one location parameter, $\mu \in \mathbb{R}$, one scale parameter $\sigma > 0$,
lower and upper bound parameters $l,u \in \mathbb{R}$, $l < u$, and point mass parameters
$L, U \ge 0$, $L + U < 1$, that assign mass to the boundary points $l$ and $u$ respectively.
Its pdf+pmf, cdf, and CRPS are

$$
f_{\mu, \sigma, l, L}^{u, U}(x) =
\begin{cases}
    0, & x < l, \\
    L, & x = l, \\
    (1 - L - U) f_{\mu, \sigma}(x), & l < x < u, \\
    U, & x = u, \\
    0, & x > u, \\
\end{cases}
$$

$$
F_{\mu, \sigma, l, L}^{u, U}(x) =
\begin{cases}
    0, & x < l, \\
    L + (1 - L - U) \frac{F_{\mu, \sigma}(x) - F_{\mu, \sigma}(l)}{F_{\mu, \sigma}(u) - F_{\mu, \sigma}(l)}, & l \le x < u, \\
    1, & x \ge u, \\
\end{cases}
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1, l, L}^{u, U}, y) &= |y - z| + uU^{2} - lL^{2} - \left( \frac{1 - L - U}{F(u) - F(l)} \right) z \left( \frac{(1 - 2L)F(u) + (1 - 2U)F(l)}{1 - L - U} \right) - \left( \frac{1 - L - U}{F(u) - F(l)} \right) (2 \log F(-z) - 2G(u)U - 2G(l)L) - \left( \frac{1 - L - U}{F(u) - F(l)} \right)^{2} (H(u) - H(l)) , \\
    \mathrm{CRPS}(F_{\mu, \sigma, l, L}^{u, U}, y) &= \sigma \mathrm{CRPS} \left( F_{0, 1, (l - \mu)/\sigma, L}^{(u - \mu)/\sigma, U}, \frac{y - \mu}{\sigma} \right),
\end{align}
$$

where $f_{\mu, \sigma}$ and $F_{\mu, \sigma}$ denote the pdf and cdf of the logistic distribution
with location parameter $\mu$ and scale parameter $\sigma$, $F = F_{0, 1}$ is the cdf of the
standard logistic distribution, and with

$$
\begin{align}
    G(x) &= xF(x) + \log F(-x), \\
    H(x) &= F(x) - xF(x)^{2} + (1 - 2F(x)) \log F(-x), \\
    z &= \max \{l, \min \{y, u\}\}.
\end{align}
$$

### Generalised truncated or censored normal distribution

The normal distribution can similarly be generalised to account for truncation, censoring, and to
allow point masses at the boundary points of its support. This *generalised normal distribution*
has one location parameter, $\mu \in \mathbb{R}$, one scale parameter $\sigma > 0$,
lower and upper bound parameters $l,u \in \mathbb{R}$, $l < u$, and point mass parameters
$L, U \ge 0$, $L + U < 1$, that assign mass to the boundary points $l$ and $u$ respectively.
Its pdf+pmf, cdf, and CRPS are

$$
f_{\mu, \sigma, l, L}^{u, U}(x) =
\begin{cases}
    0, & x < l, \\
    L, & x = l, \\
    (1 - L - U) f_{\mu, \sigma}(x), & l < x < u, \\
    U, & x = u, \\
    0, & x > u, \\
\end{cases}
$$

$$
F_{\mu, \sigma, l, L}^{u, U}(x) =
\begin{cases}
    0, & x < l, \\
    L + (1 - L - U) \frac{F_{\mu, \sigma}(x) - F_{\mu, \sigma}(l)}{F_{\mu, \sigma}(u) - F_{\mu, \sigma}(l)}, & l \le x < u, \\
    1, & x \ge u, \\
\end{cases}
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1, l, L}^{u, U}, y) &= |y - z| + uU^{2} - lL^{2} + \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right) z \left( 2 \Phi(z) - \frac{(1 - 2L)\Phi(u) + (1 - 2U)\Phi(l)}{1 - L - U} \right) + \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right) (2 \phi(z) - 2 \phi(u)U - 2 \phi(l)L) - \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right)^{2} \left( \frac{1}{\sqrt{\pi}} \right) \left(\Phi \left( u\sqrt{2} \right) - \Phi \left( l\sqrt{2} \right) \right), \\
    \mathrm{CRPS}(F_{\mu, \sigma, l, L}^{u, U}, y) &= \sigma \mathrm{CRPS} \left( F_{0, 1, (l - \mu)/\sigma, L}^{(u - \mu)/\sigma, U}, \frac{y - \mu}{\sigma} \right),
\end{align}
$$

where $f_{\mu, \sigma}$ and $F_{\mu, \sigma}$ denote the pdf and cdf of the normal distribution
with location parameter $\mu$ and scale parameter $\sigma$, and $\phi = f_{0, 1}$ and $\Phi = F_{0, 1}$
are the pdf and cdf of the standard normal distribution.


### Generalised truncated or censored Student's t distribution

The Student's t distribution can also be generalised to account for truncation, censoring, and to
allow point masses at the boundary points of its support. This *generalised t distribution*
has one location parameter, $\mu \in \mathbb{R}$, one scale parameter $\sigma > 0$,
one degrees of freedom parameter, $\nu > 0$, lower and upper bound parameters $l,u \in \mathbb{R}$,
$l < u$, and point mass parameters $L, U \ge 0$, $L + U < 1$, that assign mass to the boundary points $l$ and $u$ respectively.
Its pdf+pmf, cdf, and CRPS are

$$
f_{\mu, \sigma, \nu, l, L}^{u, U}(x) =
\begin{cases}
    0, & x < l, \\
    L, & x = l, \\
    (1 - L - U) f_{\mu, \sigma, \nu}(x), & l < x < u, \\
    U, & x = u, \\
    0, & x > u, \\
\end{cases}
$$

$$
F_{\mu, \sigma, \nu, l, L}^{u, U}(x) =
\begin{cases}
    0, & x < l, \\
    L + (1 - L - U) \frac{F_{\mu, \sigma, \nu}(x) - F_{\mu, \sigma, \nu}(l)}{F_{\mu, \sigma, \nu}(u) - F_{\mu, \sigma, \nu}(l)}, & l \le x < u, \\
    1, & x \ge u, \\
\end{cases}
$$

$$
\begin{align}
    \mathrm{CRPS}(F_{0, 1, \nu, l, L}^{u, U}, y) &= |y - z| + uU^{2} - lL^{2} + \left( \frac{1 - L - U}{F_{\nu}(u) - F_{\nu}(l)} \right) z \left( 2 F_{\nu}(z) - \frac{(1 - 2L)F_{\nu}(u) + (1 - 2U)F_{\nu}(l)}{1 - L - U} \right) - \left( \frac{1 - L - U}{F_{\nu}(u) - F_{\nu}(l)} \right) (2 G_{\nu}(z) - 2 G_{\nu}(u)U - 2 G_{\nu}(l)L) - \left( \frac{1 - L - U}{F_{\nu}(u) - F_{\nu}(l)} \right)^{2} \bar{B}_{\nu} (H_{\nu}(u) - H_{\nu}(l)), \\
    \mathrm{CRPS}(F_{\mu, \sigma, \nu, l, L}^{u, U}, y) &= \sigma \mathrm{CRPS} \left( F_{0, 1, \nu, (l - \mu)/\sigma, L}^{(u - \mu)/\sigma, U}, \frac{y - \mu}{\sigma} \right),
\end{align}
$$

where $f_{\mu, \sigma, \nu}$ and $F_{\mu, \sigma, \nu}$ denote the pdf and cdf of the Student's t distribution
with location parameter $\mu$, scale parameter $\sigma$, and degrees of freedom parameter $\nu$;
$f_{\nu} = f_{0, 1, \nu}$ and $F_{\nu} = F_{0, 1, \nu}$ are the pdf and cdf of the standard t distribution with $\nu$
degrees of freedom; and

$$
\begin{align}
    G_{\nu}(x) &= - \left( \frac{\nu + x^{2}}{\nu - 1} \right) f_{\nu}(x), \\
    H_{\nu}(x) &= \frac{1}{2} + \frac{1}{2} \text{sgn}(x) I \left(\frac{1}{2}, \nu - \frac{1}{2}, \frac{x^{2}}{\nu + x^{2}} \right), \\
    \bar{B}_{\nu} &= \left( \frac{2 \sqrt{\nu}}{\nu - 1} \right) \frac{B \left(\frac{1}{2}, \nu - \frac{1}{2} \right)}{B(\frac{1}{2}, \frac{\nu}{2})}, \\
    z &= \max \{l, \min \{y, u\}\},
\end{align}
$$

where $I(a, b, x)$ is the regularised incomplete beta function, and $B$ is the beta function.
