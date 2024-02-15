import typing as tp

from scoringrules.backend import backends
from scoringrules.core import crps

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def crps_ensemble(
    forecasts: "Array",
    observations: "ArrayLike",
    /,
    axis: int = -1,
    *,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    sorted_ensemble: bool
        Boolean indicating whether the ensemble members are already in ascending order.
        Default is False.
    estimator: str
        Indicates the CRPS estimator to be used.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    crps: ArrayLike
        The CRPS between the forecast ensemble and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.ensemble(pred, obs)
    """
    B = backends.active if backend is None else backends[backend]
    forecasts, observations = map(B.asarray, (forecasts, observations))

    if estimator not in crps.estimator_gufuncs:
        raise ValueError(
            f"{estimator} is not a valid estimator. "
            f"Must be one of {crps.estimator_gufuncs.keys()}"
        )

    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    if not sorted_ensemble and estimator not in ["nrg", "akr", "akr_circperm", "fair"]:
        forecasts = B.sort(forecasts, axis=-1)

    if backend == "numba":
        return crps.estimator_gufuncs[estimator](forecasts, observations)

    return crps.ensemble(forecasts, observations, estimator, backend=backend)


def twcrps_ensemble(
    forecasts: "Array",
    observations: "ArrayLike",
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    axis: int = -1,
    *,
    estimator: str = "pwm",
    sorted_ensemble: bool = False,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Threshold-Weighted Continuous Ranked Probability Score (twCRPS) for a finite ensemble.

    Computation is performed using the ensemble representation of the twCRPS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    $$ \mathrm{twCRPS}(F_{ens}, y) = \frac{1}{M} \sum_{m = 1}^{M} |v(x_{m}) - v(y)| - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} |v(x_{m}) - v(x_{j})|,$$

    where $F_{ens}(x) = \sum_{m=1}^{M} 1 \{ x_{m} \leq x \}/M$ is the empirical
    distribution function associated with an ensemble forecast $x_{1}, \dots, x_{M}$ with
    $M$ members, and $v$ is the chaining function used to target particular outcomes.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    v_func: tp.Callable
        Chaining function used to emphasise particular outcomes. For example, a function that
        only considers values above a certain threshold $t$ by projecting forecasts and observations
        to $[t, \inf)$.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    twcrps: ArrayLike
        The twCRPS between the forecast ensemble and obs for the chosen chaining function.

    Examples
    --------
    >>> from scoringrules import crps
    >>> twcrps.ensemble(pred, obs)
    """
    forecasts, observations = map(v_func, (forecasts, observations))
    return crps_ensemble(
        forecasts,
        observations,
        axis=axis,
        sorted_ensemble=sorted_ensemble,
        estimator=estimator,
        backend=backend,
    )


def owcrps_ensemble(
    forecasts: "Array",
    observations: "ArrayLike",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    axis: int = -1,
    *,
    estimator: tp.Literal["nrg"] = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Outcome-Weighted Continuous Ranked Probability Score (owCRPS) for a finite ensemble.

    Computation is performed using the ensemble representation of the owCRPS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    $$ \mathrm{owCRPS}(F_{ens}, y) = \frac{1}{M \bar{w}} \sum_{m = 1}^{M} |x_{m} - y|w(x_{m})w(y) - \frac{1}{2 M^{2} \bar{w}^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} |x_{m} - x_{j}|w(x_{m})w(x_{j})w(y),$$

    where $F_{ens}(x) = \sum_{m=1}^{M} 1\{ x_{m} \leq x \}/M$ is the empirical
    distribution function associated with an ensemble forecast $x_{1}, \dots, x_{M}$ with
    $M$ members, $w$ is the chosen weight function, and $\bar{w} = \sum_{m=1}^{M}w(x_{m})/M$.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    owcrps: ArrayLike
        The owCRPS between the forecast ensemble and obs for the chosen weight function.

    Examples
    --------
    >>> from scoringrules import crps
    >>> owcrps.ensemble(pred, obs)
    """
    B = backends.active if backend is None else backends[backend]

    if estimator != "nrg":
        raise ValueError(
            "Only the energy form of the estimator is available "
            "for the outcome-weighted CRPS."
        )
    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    fcts_weights, obs_weights = map(w_func, (forecasts, observations))

    if backend == "numba":
        return crps.estimator_gufuncs["ow" + estimator](
            forecasts, observations, fcts_weights, obs_weights
        )

    forecasts, observations, fcts_weights, obs_weights = map(
        B.asarray, (forecasts, observations, fcts_weights, obs_weights)
    )
    return crps.ow_ensemble(
        forecasts, observations, fcts_weights, obs_weights, backend=backend
    )


def vrcrps_ensemble(
    forecasts: "Array",
    observations: "ArrayLike",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    axis: int = -1,
    *,
    estimator: tp.Literal["nrg"] = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Vertically Re-scaled Continuous Ranked Probability Score (vrCRPS) for a finite ensemble.

    Computation is performed using the ensemble representation of the vrCRPS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    \[
    \begin{split}
        \mathrm{vrCRPS}(F_{ens}, y) = & \frac{1}{M} \sum_{m = 1}^{M} |x_{m} - y|w(x_{m})w(y) - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} |x_{m} - x_{j}|w(x_{m})w(x_{j}) \\
            & + \left( \frac{1}{M} \sum_{m = 1}^{M} |x_{m}| w(x_{m}) - |y| w(y) \right) \left( \frac{1}{M} \sum_{m = 1}^{M} w(x_{m}) - w(y) \right),
    \end{split}
    \]

    where $F_{ens}(x) = \sum_{m=1}^{M} 1 \{ x_{m} \leq x \}/M$ is the empirical
    distribution function associated with an ensemble forecast $x_{1}, \dots, x_{M}$ with
    $M$ members, $w$ is the chosen weight function, and $\bar{w} = \sum_{m=1}^{M}w(x_{m})/M$.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    vrcrps: ArrayLike
        The vrCRPS between the forecast ensemble and obs for the chosen weight function.

    Examples
    --------
    >>> from scoringrules import crps
    >>> vrcrps.ensemble(pred, obs)
    """
    B = backends.active if backend is None else backends[backend]

    if estimator != "nrg":
        raise ValueError(
            "Only the energy form of the estimator is available "
            "for the outcome-weighted CRPS."
        )
    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    fcts_weights, obs_weights = map(w_func, (forecasts, observations))

    if backend == "numba":
        return crps.estimator_gufuncs["vr" + estimator](
            forecasts, observations, fcts_weights, obs_weights
        )

    forecasts, observations, fcts_weights, obs_weights = map(
        B.asarray, (forecasts, observations, fcts_weights, obs_weights)
    )
    return crps.vr_ensemble(
        forecasts, observations, fcts_weights, obs_weights, backend=backend
    )


def crps_beta(
    shape1: "ArrayLike",
    shape2: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the beta distribution.
    
    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(F_{\alpha, \beta}, y) = (u - l)\left\{ \frac{y - l}{u - l} left( 2F_{\alpha, \beta} \left( \frac{y - l}{u - l} \right) - 1 \right) + \frac{\alpha}{\alpha + \beta} \left( 1 - 2F_{\alpha + 1, \beta} \left( \frac{y - l}{u - l} \right) - \frac{2B(2\alpha, 2\beta)}{\alpha B(\alpha, \beta)^{2}} \right) \right\}.$$

    where $F_{\alpha, \beta}$ is the beta distribution function with shape parameters $\alpha, \beta > 0$,
    and lower and upper bounds $l, u \in \R$, $l < u$.

    Parameters
    ----------
    shape1: ArrayLike
        First shape parameter of the forecast beta distribution.
    shape2: ArrayLike
        Second shape parameter of the forecast beta distribution.
    lower: ArrayLike
        Lower bound of the forecast beta distribution.
    upper: ArrayLike
        Upper bound of the forecast beta distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between Beta(shape1, shape2) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.beta(1.0, 1.0, 0.0, 1.0, 0.4)
    """
    return crps.beta(shape1, shape2, lower, upper, observation, backend=backend)


def crps_exponential(
    rate: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the exponential distribution.
    
    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(F_{\lambda}, y) = |y| - \frac{2F_{\lambda}(y)}{\lambda} + \frac{1}{2 \lambda}.$$

    where $F_{\lambda}$ is exponential distribution function with rate parameter $\lambda > 0$.

    Parameters
    ----------
    rate: ArrayLike
        Rate parameter of the forecast exponential distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between Exp(rate) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.exponential(1.0, 0.4)
    """
    return crps.exponential(rate, observation, backend=backend)


def crps_gamma(
    shape: "ArrayLike",
    rate: "ArrayLike",
    scale: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the gamma distribution.
    
    It is based on the following formulation from
    [Scheuerer and M{\"o}ller (2015)](doi: doi:10.1214/15-AOAS843):

    $$ \mathrm{CRPS}(F_{\alpha, \beta}, y) = y(2F_{\alpha, \beta}(y) - 1) - \frac{\alpha}{\beta} (2 F_{\alpha + 1, \beta}(y) - 1) - \frac{1}{\beta B(1/2, \alpha)}.$$

    where $F_{\alpha, \beta}$ is the gamma distribution function with shape parameter $\alpha > 0$
    and rate parameter $\beta > 0$ (equivalently, with scale parameter $1/\beta$).

    Parameters
    ----------
    shape: ArrayLike
        Shape parameter of the forecast gamma distribution.
    rate: ArrayLike
        Rate parameter of the forecast gamma distribution.
    scale: ArrayLike
        Scale parameter of the forecast gamma distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between Gamma(shape, rate) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.gamma(1.0, 1.0, 1.0, 0.4)
    """
    return crps.gamma(shape, rate, scale, observation, backend=backend)


def crps_laplace(
    location: "ArrayLike",
    scale: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the laplace distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(F, y) = |y - \mu| + \sigma \exp ( -| y - \mu| / \sigma) - \frac{3\sigma}{4},$$

    where $\mu$ and $\sigma > 0$ are the location and scale parameters of the Laplace distribution.

    Parameters
    ----------
    location: ArrayLike
        Location parameter of the forecast laplace distribution.
    scale: ArrayLike
        Scale parameter of the forecast laplace distribution.
    observation: ArrayLike
        Observed values.

    Returns
    -------
    crps: array_like
        The CRPS for the Laplace(location, scale) forecasts given the observations.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.laplace(0.1, 0.4, 0.0)
    """
    return crps.laplace(location, scale, observation, backend=backend)


def crps_logistic(
    mu: "ArrayLike",
    sigma: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the logistic distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(\mathcal{L}(\mu, \sigma), y) = \sigma \left\{ \omega - 2 \log F(\omega) - 1 \right\}, $$

    where $F(\omega)$ is the CDF of the standard logistic distribution at the
    normalized prediction error $\omega = \frac{y - \mu}{\sigma}$.

    Parameters
    ----------
    mu: ArrayLike
        Location parameter of the forecast logistic distribution.
    sigma: ArrayLike
        Scale parameter of the forecast logistic distribution.
    observation: ArrayLike
        Observed values.

    Returns
    -------
    crps: array_like
        The CRPS for the Logistic(mu, sigma) forecasts given the observations.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.logistic(0.1, 0.4, 0.0)
    """
    return crps.logistic(mu, sigma, observation, backend=backend)


def crps_loglogistic(
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    observation: "ArrayLike",
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the log-logistic distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(\log \mathcal{L}(\mu, \sigma), y) =
    y [2 \F_{\mu, \sigma}(y) - 1] - \exp(\mu) B(1 + \sigma, 1 - \sigma) (2 I (1 + \sigma, 1 - \sigma, F_{\mu, \sigma}(y)) + \sigma - 1).$$

    where $\F_{\mu, \sigma}$ is the CDF of the log-logistic distribution with location parameter
    $\mu$ and scale parameter $\sigma \in (0, 1)$, $B$ is the beta function, and $I$ is the
    regularised incomplete beta function.

    Parameters
    ----------
    mulog: ArrayLike
        Location parameter of the log-logistic distribution.
    sigmalog: ArrayLike
        Scale parameter of the log-logistic distribution.

    Returns
    -------
    crps: ArrayLike
        The CRPS between Loglogis(mulog, sigmalog) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.loglogistic(0.1, 0.4, 0.0)
    """
    return crps.loglogistic(mulog, sigmalog, observation, backend=backend)


def crps_lognormal(
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    observation: "ArrayLike",
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the lognormal distribution.

    It is based on the formulation introduced by
    [Baran and Lerch (2015)](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.2521)

    $$ \mathrm{CRPS}(\mathrm{log}\mathcal{N}(\mu, \sigma), y) =
    y [2 \Phi(y) - 1] - 2 \mathrm{exp}(\mu + \frac{\sigma^2}{2})
    \left[ \Phi(\omega - \sigma) + \Phi(\frac{\sigma}{\sqrt{2}}) \right]$$

    where $\Phi$ is the CDF of the standard normal distribution and
    $\omega = \frac{\mathrm{log}y - \mu}{\sigma}$.

    Note that mean and standard deviation are not the values for the distribution itself,
    but of the underlying normal distribution it is derived from.

    Parameters
    ----------
    mulog: ArrayLike
        Mean of the normal underlying distribution.
    sigmalog: ArrayLike
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    crps: ArrayLike
        The CRPS between Lognormal(mu, sigma) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.lognormal(0.1, 0.4, 0.0)
    """
    return crps.lognormal(mulog, sigmalog, observation, backend=backend)


def crps_normal(
    mu: "ArrayLike",
    sigma: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the normal distribution.

    It is based on the following formulation from
    [Gneiting et al. (2005)](https://journals.ametsoc.org/view/journals/mwre/133/5/mwr2904.1.xml):

    $$ \mathrm{CRPS}(\mathcal{N}(\mu, \sigma), y) = \sigma \Bigl\{ \omega [\Phi(ω) - 1] + 2 \phi(\omega) - \frac{1}{\sqrt{\pi}} \Bigl\},$$

    where $\Phi(ω)$ and $\phi(ω)$ are respectively the CDF and PDF of the standard normal
    distribution at the normalized prediction error $\omega = \frac{y - \mu}{\sigma}$.

    Parameters
    ----------
    mu: ArrayLike
        Mean of the forecast normal distribution.
    sigma: ArrayLike
        Standard deviation of the forecast normal distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between Normal(mu, sigma) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.normal(0.1, 0.4, 0.0)
    """
    return crps.normal(mu, sigma, observation, backend=backend)


def crps_poisson(
    mean: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the Poisson distribution.
    
    It is based on the following formulation from
    [Wei and Held (2014)](https://link.springer.com/article/10.1007/s11749-014-0380-8):

    $$ \mathrm{CRPS}(F_{\lambda}, y) = (y - \lambda) (2F_{\lambda}(y) - 1) + 2 \lambda f_{\lambda}(\lfloor y \rfloor ) - \lambda \exp (-2 \lambda) (I_{0} (2 \lambda) + I_{1} (2 \lambda))..$$

    where $F_{\lambda}$ is Poisson distribution function with mean parameter $\lambda > 0$,
    and $I_{0}$ and $I_{1}$ are modified Bessel functions of the first kind.

    Parameters
    ----------
    mean: ArrayLike
        Mean parameter of the forecast exponential distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between Pois(mean) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.poisson(1, 2)
    """
    return crps.poisson(mean, observation, backend=backend)


def crps_uniform(
    min: "ArrayLike",
    max: "ArrayLike",
    lmass: "ArrayLike",
    umass: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the uniform distribution.
    
    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(\mathcal{U}_{L}^{U}(l, u), y) = (u - l) \left\{ | \frac{y - l}{u - l} - F \left( \frac{y - l}{u - l} \right) | + F \left( \frac{y - l}{u - l} \right)^{2} (1 - L - U) - F \left( \frac{y - l}{u - l} \right) (1 - 2L) + \frac{(1 - L - U)^{2}}{3} + (1 - L)U \right\},$$

    where $\mathcal{U}_{L}^{U}(l, u)$ is the uniform distribution with lower bound $l$, 
    upper bound $u > l$, point mass $L$ on the lower bound, and point mass $U$ on the upper bound.
    We must have that $L, U \ge 0, L + U < 1$. 

    Parameters
    ----------
    min: ArrayLike
        Lower bound of the forecast uniform distribution.
    max: ArrayLike
        Upper bound of the forecast uniform distribution.
    lmass: ArrayLike
        Point mass on the lower bound of the forecast uniform distribution.
    umass: ArrayLike
        Point mass on the upper bound of the forecast uniform distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between U(min, max, lmass, umass) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.uniform(0.0, 1.0, 0.0, 0.0, 0.4)
    """
    return crps.uniform(min, max, lmass, umass, observation, backend=backend)


def crps_t(
    df: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the student's t distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(F, y) = \sigma \left\{ \omega (2 F_{\nu} (\omega) - 1) + 2 f_{\nu} \left( \frac{\nu + \omega^{2}}{\nu - 1} \right) - \frac{2 \sqrt{\nu}}{\nu - 1} \frac{B(\frac{1}{2}, \nu - \frac{1}{2})}{B(\frac{1}{2}, \frac{\nu}{2}^{2})}  \right},$$

    where $\omega = (y - \mu)/\sigma$, where $\nu > 1, \mu$, and $\sigma > 0$ are the 
    degrees of freedom, location, and scale parameters respectively of the Student's t 
    distribution, and $f_{\nu}$ and $F_{\nu}$ are the PDF and CDF of the standard Student's
    t distribution with $\nu$ degrees of freedom.
    
    Parameters
    ----------
    df: ArrayLike
        Degrees of freedom parameter of the forecast t distribution.
    location: ArrayLike
        Location parameter of the forecast t distribution.
    sigma: ArrayLike
        Scale parameter of the forecast t distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between t(df, location, scale) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.t(1, 0.1, 0.4, 0.0)
    """
    return crps.t(df, location, scale, observation, backend=backend)


def crps_tpexponential(
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the two-piece exponential distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(F_{\sigma_{1}, \sigma_{2},\mu}, y) = | y - \mu | + \frac{2 \sigma_{i}^{2}}{\sigma_{1} + \sigma_{2}} \left[ \exp \left( - \frac{| y - \mu |}{\sigma_{i}} \right) - 1 \right] + \frac{\sigma_{1}^{3} + \sigma_{2}^{3}}{2 ( \sigma_{1} + \sigma_{2}) ^{2}}, $$

    where $F_{\sigma_{1}, \sigma_{2},\mu}$ is the CDF of the two-piece exponential distribution
    with shape parameters $\sigma_{1}, \sigma_{2} > 0$ and location parameter $\mu$, with 
    $\sigma_{i} = \sigma_{1}$ if $y < \mu$ and $\sigma_{i} = \sigma_{2}$ if $y \ge \mu$.
    
    Parameters
    ----------
    scale1: ArrayLike
        First scale parameter of the forecast two-piece exponential distribution.
    scale2: ArrayLike
        Second scale parameter of the forecast two-piece exponential distribution.
    location: ArrayLike
        Location parameter of the forecast two-piece exponential distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between 2pExp(scale1, scale2, location) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.tpexponential(1.0, 1.0, 0.0, 0.4)
    """
    return crps.tpexponential(scale1, scale2, location, observation, backend=backend)


def crps_tpnorm(
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the two-piece normal distribution.

    It is based on the formulation from
    [Gneiting and Thorarinsdottir (2010)](https://arxiv.org/abs/1010.2318).
    
    Parameters
    ----------
    scale1: ArrayLike
        First scale parameter of the forecast two-piece exponential distribution.
    scale2: ArrayLike
        Second scale parameter of the forecast two-piece exponential distribution.
    location: ArrayLike
        Location parameter of the forecast two-piece exponential distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between 2pNorm(scale1, scale2, location) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.tpnorm(1.0, 1.0, 0.0, 0.4)
    """
    return crps.tpnorm(scale1, scale2, location, observation, backend=backend)


__all__ = [
    "crps_ensemble",
    "twcrps_ensemble",
    "owcrps_ensemble",
    "vrcrps_ensemble",
    "crps_beta",
    "crps_exponential",
    "crps_gamma",
    "crps_laplace",
    "crps_logistic",
    "crps_loglogistic",
    "crps_lognormal",
    "crps_normal",
    "crps_poisson",
    "crps_uniform",
    "crps_t",
    "crps_tpexponential",
    "crps_tpnorm"
]
