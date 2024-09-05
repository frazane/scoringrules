import typing as tp

from scoringrules.backend import backends
from scoringrules.core import crps, stats

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def crps_ensemble(
    observations: "ArrayLike",
    forecasts: "Array",
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
    observations: ArrayLike
        The observed values.
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
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
    >>> import scoringrules as sr
    >>> sr.crps_ensemble(obs, pred)
    """
    B = backends.active if backend is None else backends[backend]
    observations, forecasts = map(B.asarray, (observations, forecasts))

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
        return crps.estimator_gufuncs[estimator](observations, forecasts)

    return crps.ensemble(observations, forecasts, estimator, backend=backend)


def twcrps_ensemble(
    observations: "ArrayLike",
    forecasts: "Array",
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
    observations: ArrayLike
        The observed values.
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
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
    >>> import numpy as np
    >>> import scoringrules as sr
    >>>
    >>> def v_func(x):
    >>>    return np.maximum(x, -1.0)
    >>>
    >>> sr.twcrps_ensemble(obs, pred, v_func)
    """
    observations, forecasts = map(v_func, (observations, forecasts))
    return crps_ensemble(
        observations,
        forecasts,
        axis=axis,
        sorted_ensemble=sorted_ensemble,
        estimator=estimator,
        backend=backend,
    )


def crps_quantile(
    observations: "ArrayLike",
    forecasts: "Array",
    alpha: "Array",
    /,
    axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Approximate the CRPS from quantile predictions via the Pinball Loss.

    It is based on the notation in [Berrisch & Ziel, 2022](https://arxiv.org/pdf/2102.00968)

    The CRPS can be approximated as the mean pinball loss for all
    quantile forecasts $F_q$ with level $q \in Q$:

    $$\text{quantileCRPS} = \frac{2}{|Q|} \sum_{q \in Q} PB_q$$

    where the pinball loss is defined as:

    $$\text{PB}_q = \begin{cases}
        q(y - F_q) &\text{if} & y \geq F_q  \\
        (1-q)(F_q - y) &\text{else.} &  \\
    \end{cases} $$

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    alpha: Array
        The percentile levels. We expect the quantile array to match the axis (see below) of the forecast array.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    qcrps: Array
        An array of CRPS scores for each forecast, which should be averaged to get meaningful values.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_quantile(obs, fct, alpha)
    """
    B = backends.active if backend is None else backends[backend]
    observations, forecasts, alpha = map(B.asarray, (observations, forecasts, alpha))

    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    if not forecasts.shape[-1] == alpha.shape[-1]:
        raise ValueError("Expected matching length of forecasts and alpha values.")

    if B.name == "numba":
        return crps.quantile_pinball_gufunc(observations, forecasts, alpha)

    return crps.quantile_pinball(observations, forecasts, alpha, backend=backend)


def owcrps_ensemble(
    observations: "ArrayLike",
    forecasts: "Array",
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
    observations: ArrayLike
        The observed values.
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
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
    >>> import numpy as np
    >>> import scoringrules as sr
    >>>
    >>> def w_func(x):
    >>>    return (x > -1).astype(float)
    >>>
    >>> sr.owcrps_ensemble(obs, pred, w_func)
    """
    B = backends.active if backend is None else backends[backend]

    if estimator != "nrg":
        raise ValueError(
            "Only the energy form of the estimator is available "
            "for the outcome-weighted CRPS."
        )
    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    obs_weights, fct_weights = map(w_func, (observations, forecasts))

    if backend == "numba":
        return crps.estimator_gufuncs["ow" + estimator](
            observations, forecasts, obs_weights, fct_weights
        )

    observations, forecasts, obs_weights, fct_weights = map(
        B.asarray, (observations, forecasts, obs_weights, fct_weights)
    )
    return crps.ow_ensemble(
        observations, forecasts, obs_weights, fct_weights, backend=backend
    )


def vrcrps_ensemble(
    observations: "ArrayLike",
    forecasts: "Array",
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

    $$
    \begin{split}
        \mathrm{vrCRPS}(F_{ens}, y) = & \frac{1}{M} \sum_{m = 1}^{M} |x_{m} - y|w(x_{m})w(y) - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} |x_{m} - x_{j}|w(x_{m})w(x_{j}) \\
            & + \left( \frac{1}{M} \sum_{m = 1}^{M} |x_{m}| w(x_{m}) - |y| w(y) \right) \left( \frac{1}{M} \sum_{m = 1}^{M} w(x_{m}) - w(y) \right),
    \end{split}
    $$

    where $F_{ens}(x) = \sum_{m=1}^{M} 1 \{ x_{m} \leq x \}/M$ is the empirical
    distribution function associated with an ensemble forecast $x_{1}, \dots, x_{M}$ with
    $M$ members, $w$ is the chosen weight function, and $\bar{w} = \sum_{m=1}^{M}w(x_{m})/M$.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
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
    >>> import numpy as np
    >>> import scoringrules as sr
    >>>
    >>> def w_func(x):
    >>>    return (x > -1).astype(float)
    >>>
    >>> sr.vrcrps_ensemble(obs, pred, w_func)
    """
    B = backends.active if backend is None else backends[backend]

    if estimator != "nrg":
        raise ValueError(
            "Only the energy form of the estimator is available "
            "for the outcome-weighted CRPS."
        )
    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    obs_weights, fct_weights = map(w_func, (observations, forecasts))

    if backend == "numba":
        return crps.estimator_gufuncs["vr" + estimator](
            observations, forecasts, obs_weights, fct_weights
        )

    observations, forecasts, obs_weights, fct_weights = map(
        B.asarray, (observations, forecasts, obs_weights, fct_weights)
    )
    return crps.vr_ensemble(
        observations, forecasts, obs_weights, fct_weights, backend=backend
    )


def crps_beta(
    observation: "ArrayLike",
    a: "ArrayLike",
    b: "ArrayLike",
    /,
    lower: "ArrayLike" = 0.0,
    upper: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the beta distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$
    \mathrm{CRPS}(F_{\alpha, \beta}, y) = (u - l)\left\{ \frac{y - l}{u - l}
    \left( 2F_{\alpha, \beta} \left( \frac{y - l}{u - l} \right) - 1 \right)
    + \frac{\alpha}{\alpha + \beta} \left( 1 - 2F_{\alpha + 1, \beta}
    \left( \frac{y - l}{u - l} \right)
    - \frac{2B(2\alpha, 2\beta)}{\alpha B(\alpha, \beta)^{2}} \right) \right\}
    $$

    where $F_{\alpha, \beta}$ is the beta distribution function with shape parameters
    $\alpha, \beta > 0$, and lower and upper bounds $l, u \in \R$, $l < u$.

    Parameters
    ----------
    observation:
        The observed values.
    a:
        First shape parameter of the forecast beta distribution.
    b:
        Second shape parameter of the forecast beta distribution.
    lower:
        Lower bound of the forecast beta distribution.
    upper:
        Upper bound of the forecast beta distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between Beta(a, b) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_beta(0.3, 0.7, 1.1)
    0.0850102437
    """
    return crps.beta(observation, a, b, lower, upper, backend=backend)


def crps_binomial(
    observation: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the binomial distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$
    \mathrm{CRPS}(F_{n, p}, y) = 2 \sum_{x = 0}^{n} f_{n,p}(x) (1\{y < x\}
    - F_{n,p}(x) + f_{n,p}(x)/2) (x - y),
    $$

    where $f_{n, p}$ and $F_{n, p}$ are the PDF and CDF of the binomial distribution
    with size parameter $n = 0, 1, 2, ...$ and probability parameter $p \in [0, 1]$.

    Parameters
    ----------
    observation:
        The observed values as an integer or array of integers.
    n:
        Size parameter of the forecast binomial distribution as an integer or array of integers.
    prob:
        Probability parameter of the forecast binomial distribution as a float or array of floats.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between Binomial(n, prob) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_binomial(4, 10, 0.5)
    0.5955715179443359
    """
    return crps.binomial(observation, n, prob, backend=backend)


def crps_exponential(
    observation: "ArrayLike",
    rate: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the exponential distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$\mathrm{CRPS}(F_{\lambda}, y) = |y| - \frac{2F_{\lambda}(y)}{\lambda} + \frac{1}{2 \lambda},$$

    where $F_{\lambda}$ is exponential distribution function with rate parameter $\lambda > 0$.

    Parameters
    ----------
    observation:
        The observed values.
    rate:
        Rate parameter of the forecast exponential distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between Exp(rate) and obs.

    Examples
    --------
    ```pycon
    >>> import scoringrules as sr
    >>> import numpy as np
    >>> sr.crps_exponential(0.8, 3.0)
    0.360478635526275
    >>> sr.crps_exponential(np.array([0.8, 0.9]), np.array([3.0, 2.0]))
    array([0.36047864, 0.24071795])
    ```
    """
    return crps.exponential(observation, rate, backend=backend)


def crps_exponentialM(
    observation: "ArrayLike",
    /,
    mass: "ArrayLike" = 0.0,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the standard exponential distribution with a point mass at the boundary.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(F_{M}, y) = |y| - 2 (1 - M) F(y) + \frac{(1 - M)**2}{2}, $$

    $$ \mathrm{CRPS}(F_{M, \mu, \sigma}, y) =
    \sigma \mathrm{CRPS} \left( F_{M}, \frac{y - \mu}{\sigma} \right), $$

    where $F_{M, \mu, \sigma}$ is standard exponential distribution function
    generalised using a location parameter $\mu$ and scale parameter $\sigma < 0$
    and a point mass $M \in [0, 1]$ at $\mu$, $F_{M} = F_{M, 0, 1}$, and

    $$ F(y) = 1 - \exp(-y) $$

    for $y \geq 0$, and 0 otherwise.

    Parameters
    ----------
    observation:
        The observed values.
    mass:
        Mass parameter of the forecast exponential distribution.
    location:
        Location parameter of the forecast exponential distribution.
    scale:
        Scale parameter of the forecast exponential distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between obs and ExpM(mass, location, scale).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_exponentialM(0.4, 0.2, 0.0, 1.0)
    """
    return crps.exponentialM(observation, mass, location, scale, backend=backend)


def crps_gamma(
    observation: "ArrayLike",
    shape: "ArrayLike",
    /,
    rate: "ArrayLike | None" = None,
    *,
    scale: "ArrayLike | None" = None,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the gamma distribution.

    It is based on the following formulation from
    [Scheuerer and Möller (2015)](doi: doi:10.1214/15-AOAS843):

    $$ \mathrm{CRPS}(F_{\alpha, \beta}, y) = y(2F_{\alpha, \beta}(y) - 1)
    - \frac{\alpha}{\beta} (2 F_{\alpha + 1, \beta}(y) - 1)
    - \frac{1}{\beta B(1/2, \alpha)}. $$

    where $F_{\alpha, \beta}$ is gamma distribution function with shape
    parameter $\alpha > 0$ and rate parameter $\beta > 0$ (equivalently,
    with scale parameter $1/\beta$).

    Parameters
    ----------
    observation:
        The observed values.
    shape:
        Shape parameter of the forecast gamma distribution.
    rate:
        Rate parameter of the forecast rate distribution.
    scale:
        Scale parameter of the forecast scale distribution, where `scale = 1 / rate`.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between obs and Gamma(shape, rate).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gamma(0.2, 1.1, 0.1)
    5.503536008961291

    Raises
    ------
    ValueError
        If both `rate` and `scale` are provided, or if neither is provided.
    """
    if (scale is None and rate is None) or (scale is not None and rate is not None):
        raise ValueError(
            "Either `rate` or `scale` must be provided, but not both or neither."
        )

    if rate is None:
        rate = 1.0 / scale

    return crps.gamma(observation, shape, rate, backend=backend)


def crps_gev(
    observation: "ArrayLike",
    shape: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the generalised extreme value (GEV) distribution.

    It is based on the following formulation from
    [Friederichs and Thorarinsdottir (2012)](doi/10.1002/env.2176):

    $$
    \text{CRPS}(F_{\xi, \mu, \sigma}, y) =
    \sigma \cdot \text{CRPS}(F_{\xi}, \frac{y - \mu}{\sigma})
    $$

    Special cases are handled as follows:

    - For $\xi = 0$:

    $$
    \text{CRPS}(F_{\xi}, y) = -y - 2\text{Ei}(\log F_{\xi}(y)) + C - \log 2
    $$

    - For $\xi \neq 0$:

    $$
    \text{CRPS}(F_{\xi}, y) = y(2F_{\xi}(y) - 1) - 2G_{\xi}(y)
    - \frac{1 - (2 - 2^{\xi}) \Gamma(1 - \xi)}{\xi}
    $$

    where $C$ is the Euler-Mascheroni constant, $\text{Ei}$ is the exponential
    integral, and $\Gamma$ is the gamma function. The GEV cumulative distribution
    function $F_{\xi}$ and the auxiliary function $G_{\xi}$ are defined as:

    - For $\xi = 0$:

    $$
    F_{\xi}(x) = \exp(-\exp(-x))
    $$

    - For $\xi \neq 0$:

    $$
    F_{\xi}(x) =
    \begin{cases}
    0, & x \leq \frac{1}{\xi} \\
    \exp(-(1 + \xi x)^{-1/\xi}), & x > \frac{1}{\xi}
    \end{cases}
    $$

    $$
    G_{\xi}(x) =
    \begin{cases}
    0, & x \leq \frac{1}{\xi} \\
    \frac{F_{\xi}(x)}{\xi} + \frac{\Gamma_u(1-\xi, -\log F_{\xi}(x))}{\xi}, & x > \frac{1}{\xi}
    \end{cases}
    $$

    Parameters
    ----------
    observation:
        The observed values.
    shape:
        Shape parameter of the forecast GEV distribution.
    location:
        Location parameter of the forecast GEV distribution.
    scale:
        Scale parameter of the forecast GEV distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.


    Returns
    -------
    score:
        The CRPS between obs and GEV(shape, location, scale).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gev(0.3, 0.1)
    0.2924712413052034
    """
    return crps.gev(observation, shape, location, scale, backend=backend)


def crps_gpd(
    observation: "ArrayLike",
    shape: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    mass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the generalised pareto distribution (GPD).

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

       
    $$
    \mathrm{CRPS}(F_{M, \xi}, y) =
    |y| - \frac{2 (1 - M)}{1 - \xi} \left( 1 - (1 - F_{\xi}(y))^{1 - \xi} \right)
    + \frac{(1 - M)^{2}}{2 - \xi},
    $$

    $$
    \mathrm{CRPS}(F_{M, \xi, \mu, \sigma}, y) =
    \sigma \mathrm{CRPS} \left( F_{M, \xi}, \frac{y - \mu}{\sigma} \right),
    $$

    where $F_{M, \xi, \mu, \sigma}$ is the GPD distribution function with shape
    parameter $\xi < 1$, location parameter $\mu$, scale parameter $\sigma > 0$,
    and point mass $M \in [0, 1]$ at the lower boundary. $F_{M, \xi} = F_{M, \xi, 0, 1}$.

    Parameters
    ----------
    observation:
        The observed values.
    shape:
        Shape parameter of the forecast GPD distribution.
    location:
        Location parameter of the forecast GPD distribution.
    scale:
        Scale parameter of the forecast GPD distribution.
    mass:
        Mass parameter at the lower boundary of the forecast GPD distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between obs and GPD(shape, location, scale, mass).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gpd(0.3, 0.9)
    0.6849331901197213
    """
    return crps.gpd(observation, shape, location, scale, mass, backend=backend)


def crps_gtclogistic(
    observation: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    /,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    lmass: "ArrayLike" = 0.0,
    umass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the generalised truncated and censored logistic distribution.

    $$ \mathrm{CRPS}(F_{l, L}^{u, U}, y) = |y - z| + uU^{2} - lL^{2} - \left( \frac{1 - L - U}{F(u) - F(l)} \right) z \left( \frac{(1 - 2L) F(u) + (1 - 2U) F(l)}{1 - L - U} \right) - \left( \frac{1 - L - U}{F(u) - F(l)} \right) \left( 2 \logF(-z) - 2G(u)U - 2 G(l)L \right) - \left( \frac{1 - L - U}{F(u) - F(l)} \right)^{2} \left( H(u) - H(l) \right), $$
    
    $$ \mathrm{CRPS}(F_{l, L, \mu, \sigma}^{u, U}, y) = \sigma \mathrm{CRPS}(F_{(l - \mu)/\sigma, L}^{(u - \mu)/\sigma, U}, \frac{y - \mu}{\sigma}), $$
    
    $$G(x) = xF(x) + \log F(-x),$$
    
    $$H(x) = F(x) - xF(x)^{2} + (1 - 2F(x))\log F(-x),$$

    where $F$ is the CDF of the standard logistic distribution, $F_{l, L, \mu, \sigma}^{u, U}$ 
    is the CDF of the logistic distribution truncated below at $l$ and above at $u$, 
    with point masses $L, U > 0$ at the lower and upper boundaries, respectively, and 
    location and scale parameters $\mu$ and $\sigma > 0$.
    
    Parameters
    ----------
    observation: ArrayLike
        The observed values.
    location: ArrayLike
        Location parameter of the forecast distribution.
    scale: ArrayLike
        Scale parameter of the forecast distribution.
    lower: ArrayLike
        Lower boundary of the truncated forecast distribution.
    upper: ArrayLike
        Upper boundary of the truncated forecast distribution.
    lmass: ArrayLike
        Point mass assigned to the lower boundary of the forecast distribution.
    umass: ArrayLike
        Point mass assigned to the upper boundary of the forecast distribution.

    Returns
    -------
    crps: array_like
        The CRPS between gtcLogistic(location, scale, lower, upper, lmass, umass) and obs.
        
    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gtclogistic(0.0, 0.1, 0.4, -1.0, 1.0, 0.1, 0.1)
    """
    return crps.gtclogistic(observation, location, scale, lower, upper, lmass, umass, backend=backend)


def crps_tlogistic(
    observation: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    /,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the truncated logistic distribution.

    It is based on the formulation for the generalised truncated and censored logistic 
    distribution with lmass and umass set to zero.
    
    Parameters
    ----------
    observation: ArrayLike
        The observed values.
    location: ArrayLike
        Location parameter of the forecast distribution.
    scale: ArrayLike
        Scale parameter of the forecast distribution.
    lower: ArrayLike
        Lower boundary of the truncated forecast distribution.
    upper: ArrayLike
        Upper boundary of the truncated forecast distribution.

    Returns
    -------
    crps: array_like
        The CRPS between tLogistic(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_tlogistic(0.0, 0.1, 0.4, -1.0, 1.0)
    """
    return crps.gtclogistic(observation, location, scale, lower, upper, 0.0, 0.0, backend=backend)


def crps_clogistic(
    observation: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    /,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the censored logistic distribution.

    It is based on the formulation for the generalised truncated and censored logistic distribution with 
    lmass and umass set to the tail probabilities of the predictive distribution.

    Parameters
    ----------
    observation: ArrayLike
        The observed values.
    location: ArrayLike
        Location parameter of the forecast distribution.
    scale: ArrayLike
        Scale parameter of the forecast distribution.
    lower: ArrayLike
        Lower boundary of the truncated forecast distribution.
    upper: ArrayLike
        Upper boundary of the truncated forecast distribution.

    Returns
    -------
    crps: array_like
        The CRPS between cLogistic(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_clogistic(0.0, 0.1, 0.4, -1.0, 1.0)
    """
    lmass = stats._logis_cdf((lower - location) / scale)
    umass = 1 - stats._logis_cdf((upper - location) / scale)
    return crps.gtclogistic(observation, location, scale, lower, upper, lmass, umass, backend=backend)


def crps_gtcnormal(
    observation: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    /,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    lmass: "ArrayLike" = 0.0,
    umass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    
    r"""Compute the closed form of the CRPS for the generalised truncated and censored normal distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$ \mathrm{CRPS}(F_{l, L}^{u, U}, y) = |y - z| + uU^{2} - lL^{2} + \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right) z \left( 2 \Phi(z) - \frac{(1 - 2L) \Phi(u) + (1 - 2U) \Phi(l)}{1 - L - U} \right) + \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right) \left( 2 \phi(z) - 2 \phi(u)U - 2 \phi(l)L \right) - \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right)^{2} \left( \frac{1}{\sqrt{\pi}} \right) \left( \Phi(u \sqrt{2}) - \Phi(l \sqrt{2}) \right), $$
    
    $$ \mathrm{CRPS}(F_{l, L, \mu, \sigma}^{u, U}, y) = \sigma \mathrm{CRPS}(F_{(l - \mu)/\sigma, L}^{(u - \mu)/\sigma, U}, \frac{y - \mu}{\sigma}), $$

    where $\Phi$ and $\phi$ are respectively the CDF and PDF of the standard normal
    distribution, $F_{l, L, \mu, \sigma}^{u, U}$ is the CDF of the normal distribution 
    truncated below at $l$ and above at $u$, with point masses $L, U > 0$ at the lower and upper
    boundaries, respectively, and location and scale parameters $\mu$ and $\sigma > 0$.
    $F_{l, L}^{u, U} = F_{l, L, 0, 1}^{u, U}$.

    Examples
    --------
    >>> import scoring rules as sr
    >>> sr.crps_gtcnormal(0.0, 0.1, 0.4, -1.0, 1.0, 0.1, 0.1)
    """
    return crps.gtcnormal(observation, location, scale, lower, upper, lmass, umass, backend=backend)


def crps_tnormal(
    observation: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    /,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the truncated normal distribution.

    It is based on the formulation for the generalised truncated and censored normal distribution with 
    distribution with lmass and umass set to zero.
    
    Parameters
    ----------
    observation: ArrayLike
        The observed values.
    location: ArrayLike
        Location parameter of the forecast distribution.
    scale: ArrayLike
        Scale parameter of the forecast distribution.
    lower: ArrayLike
        Lower boundary of the truncated forecast distribution.
    upper: ArrayLike
        Upper boundary of the truncated forecast distribution.

    Returns
    -------
    crps: array_like
        The CRPS between tNormal(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_tnormal(0.0, 0.1, 0.4, -1.0, 1.0)
    """
    return crps.gtcnormal(observation, location, scale, lower, upper, 0.0, 0.0, backend=backend)


def crps_cnormal(
    observation: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    /,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the censored normal distribution.

    It is based on the formulation for the generalised truncated and censored normal distribution with 
    lmass and umass set to the tail probabilities of the predictive distribution.

    Parameters
    ----------
    observation: ArrayLike
        The observed values.
    location: ArrayLike
        Location parameter of the forecast distribution.
    scale: ArrayLike
        Scale parameter of the forecast distribution.
    lower: ArrayLike
        Lower boundary of the truncated forecast distribution.
    upper: ArrayLike
        Upper boundary of the truncated forecast distribution.

    Returns
    -------
    crps: array_like
        The CRPS between cNormal(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_cnormal(0.0, 0.1, 0.4, -1.0, 1.0)
    """
    lmass = stats._norm_cdf((lower - location) / scale)
    umass = 1 - stats._norm_cdf((upper - location) / scale)
    return crps.gtcnormal(observation, location, scale, lower, upper, lmass, umass, backend=backend)


def crps_hypergeometric(
    observation: "ArrayLike",
    m: "ArrayLike",
    n: "ArrayLike",
    k: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the hypergeometric distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$
    \mathrm{CRPS}(F_{m, n, k}, y) = 2 \sum_{x = 0}^{n} f_{m,n,k}(x) (1\{y < x\}
    - F_{m,n,k}(x) + f_{m,n,k}(x)/2) (x - y),
    $$

    where $f_{m, n, k}$ and $F_{m, n, k}$ are the PDF and CDF of the hypergeometric
    distribution with population parameters $m,n = 0, 1, 2, ...$ and size parameter
    $k = 0, ..., m + n$.

    Parameters
    ----------
    observation:
        The observed values.
    m:
        Number of success states in the population.
    n:
        Number of failure states in the population.
    k:
        Number of draws, without replacement. Must be in 0, 1, ..., m + n.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between obs and Hypergeometric(m, n, k).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_hypergeometric(5, 7, 13, 12)
    0.44697415547610597
    """
    return crps.hypergeometric(observation, m, n, k, backend=backend)


def crps_laplace(
    observation: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the laplace distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$
    \mathrm{CRPS}(F, y) = |y - \mu|
    + \sigma \exp ( -| y - \mu| / \sigma) - \frac{3\sigma}{4},
    $$

    where $\mu$ and $\sigma > 0$ are the location and scale parameters
    of the Laplace distribution.

    Parameters
    ----------
    observation:
        Observed values.
    location:
        Location parameter of the forecast laplace distribution.
    scale:
        Scale parameter of the forecast laplace distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between obs and Laplace(location, scale).

    >>> sr.crps_laplace(0.3, 0.1, 0.2)
    0.12357588823428847
    """
    return crps.laplace(observation, location, scale, backend=backend)


def crps_logistic(
    observation: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
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
    observations: ArrayLike
        Observed values.
    mu: ArrayLike
        Location parameter of the forecast logistic distribution.
    sigma: ArrayLike
        Scale parameter of the forecast logistic distribution.

    Returns
    -------
    crps: array_like
        The CRPS for the Logistic(mu, sigma) forecasts given the observations.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_logistic(0.0, 0.4, 0.1)
    0.30363
    """
    return crps.logistic(observation, mu, sigma, backend=backend)


def crps_loglaplace(
    observation: "ArrayLike",
    locationlog: "ArrayLike",
    scalelog: "ArrayLike",
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the log-Laplace distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$
    \mathrm{CRPS}(F_{\mu, \sigma}, y) = y (2 F_{\mu, \sigma}(y) - 1)
    + \exp(\mu) \left( \frac{\sigma}{4 - \sigma^{2}} + A(y) \right),
    $$

    where $F_{\mu, \sigma}$ is the CDF of the log-laplace distribution with location
    parameter $\mu$ and scale parameter $\sigma \in (0, 1)$, and

    $$ A(y) = \frac{1}{1 + \sigma} \left( 1 - (2 F_{\mu, \sigma}(y) - 1)^{1 + \sigma} \right), $$

    if $y < \exp{\mu}$, and

    $$ A(y) = \frac{-1}{1 - \sigma} \left( 1 - (2 (1 - F_{\mu, \sigma}(y)))^{1 - \sigma} \right), $$

    if $y \ge \exp{\mu}$.

    Parameters
    ----------
    observation:
        Observed values.
    locationlog:
        Location parameter of the forecast log-laplace distribution.
    scalelog:
        Scale parameter of the forecast log-laplace distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The CRPS between obs and Loglaplace(locationlog, scalelog).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_loglaplace(3.0, 0.1, 0.9)
    1.162020513653791
    """
    return crps.loglaplace(observation, locationlog, scalelog, backend=backend)


def crps_loglogistic(
    observation: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the log-logistic distribution.

    It is based on the following formulation from
    [Jordan et al. (2019)](https://www.jstatsoft.org/article/view/v090i12):

    $$
    \text{CRPS}(y, F_{\mu,\sigma}) = y \left( 2F_{\mu,\sigma}(y) - 1 \right)
    - 2 \exp(\mu) I\left(F_{\mu,\sigma}(y); 1 + \sigma, 1 - \sigma\right) \\
    + \exp(\mu)(1 - \sigma) B(1 + \sigma, 1 - \sigma)
    $$

    where \( F_{\mu,\sigma}(x) \) is the cumulative distribution function (CDF) of
    the log-logistic distribution, defined as:

    $$
    F_{\mu,\sigma}(x) =
    \begin{cases}
    0, & x \leq 0 \\
    \left( 1 + \exp\left(-\frac{\log x - \mu}{\sigma}\right) \right)^{-1}, & x > 0
    \end{cases}
    $$

    $B$ is the beta function, and $I$ is the regularised incomplete beta function.

    Parameters
    ----------
    observation:
        The observed values.
    mulog:
        Location parameter of the log-logistic distribution.
    sigmalog:
        Scale parameter of the log-logistic distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.


    Returns
    -------
    score:
        The CRPS between obs and Loglogis(mulog, sigmalog).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_loglogistic(3.0, 0.1, 0.9)
    1.1329527730161177
    """
    return crps.loglogistic(observation, mulog, sigmalog, backend=backend)


def crps_lognormal(
    observation: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
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
    observation:
        The observed values.
    mulog:
        Mean of the normal underlying distribution.
    sigmalog:
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    crps: ArrayLike
        The CRPS between Lognormal(mu, sigma) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_lognormal(0.1, 0.4, 0.0)
    """
    return crps.lognormal(observation, mulog, sigmalog, backend=backend)


def crps_normal(
    observation: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the normal distribution.

    It is based on the following formulation from
    [Geiting et al. (2005)](https://journals.ametsoc.org/view/journals/mwre/133/5/mwr2904.1.xml):

    $$ \mathrm{CRPS}(\mathcal{N}(\mu, \sigma), y) = \sigma \Bigl\{ \omega [\Phi(ω) - 1] + 2 \phi(\omega) - \frac{1}{\sqrt{\pi}} \Bigl\},$$

    where $\Phi(ω)$ and $\phi(ω)$ are respectively the CDF and PDF of the standard normal
    distribution at the normalized prediction error $\omega = \frac{y - \mu}{\sigma}$.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    mu: ArrayLike
        Mean of the forecast normal distribution.
    sigma: ArrayLike
        Standard deviation of the forecast normal distribution.

    Returns
    -------
    crps: array_like
        The CRPS between Normal(mu, sigma) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_normal(0.1, 0.4, 0.0)
    """
    return crps.normal(observation, mu, sigma, backend=backend)


__all__ = [
    "crps_ensemble",
    "twcrps_ensemble",
    "owcrps_ensemble",
    "vrcrps_ensemble",
    "crps_quantile",
    "crps_beta",
    "crps_binomial",
    "crps_exponential",
    "crps_exponentialM",
    "crps_gamma",
    "crps_gev",
    "crps_gpd",
<<<<<<< HEAD
    "crps_gtclogistic",
    "crps_tlogistic",
    "crps_clogistic",
=======
    "crps_gtcnormal",
    "crps_tnormal",
    "crps_cnormal",
>>>>>>> main
    "crps_hypergeometric",
    "crps_laplace",
    "crps_logistic",
    "crps_loglaplace",
    "crps_loglogistic",
    "crps_lognormal",
    "crps_normal",
]
