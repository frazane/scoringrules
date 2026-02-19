import typing as tp

from scoringrules.backend import backends
from scoringrules.core import logarithmic
from scoringrules.core.utils import univariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def logs_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    m_axis: int = -1,
    *,
    bw: "ArrayLike" = None,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Logarithmic score for a finite ensemble via kernel density estimation.

    Gaussian kernel density estimation is used to convert the finite ensemble to a
    mixture of normal distributions, with the component distributions centred at each
    ensemble member, with scale equal to the bandwidth parameter 'bw'.

    The log score for the ensemble forecast is then the log score for the mixture of
    normal distributions.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    bw : array_like
        The bandwidth parameter for each forecast ensemble. If not given, estimated using
        Silverman's rule of thumb.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score : array_like
        The LS between the forecast ensemble and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_ensemble(obs, pred)
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = univariate_array_check(obs, fct, m_axis, backend=backend)

    M = fct.shape[-1]  # number of ensemble members

    # Silverman's rule of thumb for estimating the bandwidth parameter
    if bw is None:
        sigmahat = B.std(fct, axis=-1)
        q75 = B.quantile(fct, 0.75, axis=-1)
        q25 = B.quantile(fct, 0.25, axis=-1)
        iqr = q75 - q25
        bw = 1.06 * B.minimum(sigmahat, iqr / 1.34) * (M ** (-1 / 5))
    bw = B.stack([bw] * M, axis=-1)

    w = B.ones(fct.shape) / M

    return logarithmic.mixnorm(obs, fct, bw, w, backend=backend)


def clogs_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    a: "ArrayLike" = float("-inf"),
    b: "ArrayLike" = float("inf"),
    m_axis: int = -1,
    *,
    bw: "ArrayLike" = None,
    cens: bool = True,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the conditional and censored likelihood score for an ensemble forecast.

    The conditional and censored likelihood scores are introduced by
    [Diks et al. (2011)](https://doi.org/10.1016/j.jeconom.2011.04.001):

    The weight function is an indicator function of the form :math:`w(z) = 1\{a < z < b\}`.

    The ensemble forecast is converted to a mixture of normal distributions using Gaussian
    kernel density estimation. The score is then calculated for this smoothed distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    a : array_like
        The lower bound in the weight function.
    b : array_like
        The upper bound in the weight function.
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    bw : array_like
        The bandwidth parameter for each forecast ensemble. If not given, estimated using
        Silverman's rule of thumb.
    cens : Boolean
        Boolean specifying whether to return the conditional ('cens = False') or the censored
        likelihood score ('cens = True').
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score : array_like
        The CoLS or CeLS between the forecast ensemble and obs for the chosen weight parameters.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.clogs_ensemble(obs, pred, -1.0, 1.0)
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = univariate_array_check(obs, fct, m_axis, backend=backend)

    M = fct.shape[-1]  # number of ensemble members

    # Silverman's rule of thumb for estimating the bandwidth parameter
    if bw is None:
        sigmahat = B.std(fct, axis=-1)
        q75 = B.quantile(fct, 0.75, axis=-1)
        q25 = B.quantile(fct, 0.25, axis=-1)
        iqr = q75 - q25
        bw = 1.06 * B.minimum(sigmahat, iqr / 1.34) * (M ** (-1 / 5))
    bw = B.stack([bw] * M, axis=-1)

    return logarithmic.clogs_ensemble(
        obs,
        fct,
        a=a,
        b=b,
        bw=bw,
        cens=cens,
        backend=backend,
    )


def logs_beta(
    obs: "ArrayLike",
    /,
    a: "ArrayLike",
    b: "ArrayLike",
    lower: "ArrayLike" = 0.0,
    upper: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the beta distribution.

    This score is equivalent to the negative log likelihood of the beta distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    a : array_like
        First shape parameter of the forecast beta distribution.
    b : array_like
        Second shape parameter of the forecast beta distribution.
    lower : array_like
        Lower bound of the forecast beta distribution.
    upper : array_like
        Upper bound of the forecast beta distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between Beta(a, b) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_beta(0.3, 0.7, 1.1)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        a, b, lower, upper = map(B.asarray, (a, b, lower, upper))
        if B.any(a <= 0):
            raise ValueError(
                "`a` contains non-positive entries. The shape parameters of the Beta distribution must be positive."
            )
        if B.any(b <= 0):
            raise ValueError(
                "`b` contains non-positive entries. The shape parameters of the Beta distribution must be positive."
            )
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    return logarithmic.beta(obs, a, b, lower, upper, backend=backend)


def logs_binomial(
    obs: "ArrayLike",
    /,
    n: "ArrayLike",
    prob: "ArrayLike",
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the binomial distribution.

    This score is equivalent to the negative log likelihood of the binomial distribution

    Parameters
    ----------
    obs : array_like
        The observed values.
    n : array_like
        Size parameter of the forecast binomial distribution as an integer or array of integers.
    prob : array_like
        Probability parameter of the forecast binomial distribution as a float or array of floats.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between Binomial(n, prob) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_binomial(4, 10, 0.5)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        prob = B.asarray(prob)
        if B.any(prob < 0) or B.any(prob > 1):
            raise ValueError("`prob` contains values outside the range [0, 1].")
    return logarithmic.binomial(obs, n, prob, backend=backend)


def logs_exponential(
    obs: "ArrayLike",
    /,
    rate: "ArrayLike",
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the exponential distribution.

    This score is equivalent to the negative log likelihood of the exponential distribution

    Parameters
    ----------
    obs : array_like
        The observed values.
    rate : array_like
        Rate parameter of the forecast exponential distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between Exp(rate) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_exponential(0.8, 3.0)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        rate = B.asarray(rate)
        if B.any(rate <= 0):
            raise ValueError(
                "`rate` contains non-positive entries. The rate parameter of the exponential distribution must be positive."
            )
    return logarithmic.exponential(obs, rate, backend=backend)


def logs_exponential2(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the exponential distribution with location and scale parameters.

    This score is equivalent to the negative log likelihood of the exponential distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    location : array_like
        Location parameter of the forecast exponential distribution.
    scale : array_like
        Scale parameter of the forecast exponential distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between obs and Exp2(location, scale).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_exponential2(0.2, 0.0, 1.0)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale = B.asarray(scale)
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the exponential distribution must be positive."
            )
    return logarithmic.exponential2(obs, location, scale, backend=backend)


def logs_2pexponential(
    obs: "ArrayLike",
    /,
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the two-piece exponential distribution.

    This score is equivalent to the negative log likelihood of the two-piece exponential distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    scale1 : array_like
        First scale parameter of the forecast two-piece exponential distribution.
    scale2 : array_like
        Second scale parameter of the forecast two-piece exponential distribution.
    location : array_like
        Location parameter of the forecast two-piece exponential distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between 2pExp(sigma1, sigma2, location) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_2pexponential(0.8, 3.0, 1.4, 0.0)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale1, scale2 = map(B.asarray, (scale1, scale2))
        if B.any(scale1 <= 0):
            raise ValueError(
                "`scale1` contains non-positive entries. The scale parameters of the two-piece exponential distribution must be positive."
            )
        if B.any(scale2 <= 0):
            raise ValueError(
                "`scale2` contains non-positive entries. The scale parameters of the two-piece exponential distribution must be positive."
            )
    return logarithmic.twopexponential(obs, scale1, scale2, location, backend=backend)


def logs_gamma(
    obs: "ArrayLike",
    /,
    shape: "ArrayLike",
    rate: "ArrayLike | None" = None,
    *,
    scale: "ArrayLike | None" = None,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the gamma distribution.

    This score is equivalent to the negative log likelihood of the gamma distribution

    Parameters
    ----------
    obs : array_like
        The observed values.
    shape : array_like
        Shape parameter of the forecast gamma distribution.
    rate : array_like
        Rate parameter of the forecast gamma distribution.
    scale : array_like
        Scale parameter of the forecast gamma distribution, where `scale = 1 / rate`.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between obs and Gamma(shape, rate).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_gamma(0.2, 1.1, 0.1)

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

    if check_pars:
        B = backends.active if backend is None else backends[backend]
        shape, rate = map(B.asarray, (shape, rate))
        if B.any(shape <= 0):
            raise ValueError(
                "`shape` contains non-positive entries. The shape parameter of the gamma distribution must be positive."
            )
        if B.any(rate <= 0):
            raise ValueError(
                "`rate` or `scale` contains non-positive entries. The rate and scale parameters of the gamma distribution must be positive."
            )

    return logarithmic.gamma(obs, shape, rate, backend=backend)


def logs_gev(
    obs: "ArrayLike",
    /,
    shape: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the generalised extreme value (GEV) distribution.

    This score is equivalent to the negative log likelihood of the GEV distribution

    Parameters
    ----------
    obs : array_like
        The observed values.
    shape : array_like
        Shape parameter of the forecast GEV distribution.
    location : array_like
        Location parameter of the forecast GEV distribution.
    scale : array_like
        Scale parameter of the forecast GEV distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between obs and GEV(shape, location, scale).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_gev(0.3, 0.1)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, shape = map(B.asarray, (scale, shape))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the GEV distribution must be positive."
            )
    return logarithmic.gev(obs, shape, location, scale, backend=backend)


def logs_gpd(
    obs: "ArrayLike",
    /,
    shape: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the generalised Pareto distribution (GPD).

    This score is equivalent to the negative log likelihood of the GPD


    Parameters
    ----------
    obs : array_like
        The observed values.
    shape : array_like
        Shape parameter of the forecast GPD distribution.
    location : array_like
        Location parameter of the forecast GPD distribution.
    scale : array_like
        Scale parameter of the forecast GPD distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between obs and GPD(shape, location, scale).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_gpd(0.3, 0.9)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, shape = map(B.asarray, (scale, shape))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the GPD distribution must be positive. `nan` is returned in these places."
            )
    return logarithmic.gpd(obs, shape, location, scale, backend=backend)


def logs_hypergeometric(
    obs: "ArrayLike",
    /,
    m: "ArrayLike",
    n: "ArrayLike",
    k: "ArrayLike",
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the hypergeometric distribution.

    This score is equivalent to the negative log likelihood of the hypergeometric distribution

    Parameters
    ----------
    obs : array_like
        The observed values.
    m : array_like
        Number of success states in the population.
    n : array_like
        Number of failure states in the population.
    k : array_like
        Number of draws, without replacement. Must be in 0, 1, ..., m + n.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between obs and Hypergeometric(m, n, k).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_hypergeometric(5, 7, 13, 12)
    """
    # TODO: add check that m,n,k are integers
    return logarithmic.hypergeometric(obs, m, n, k, backend=backend)


def logs_laplace(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the Laplace distribution.

    This score is equivalent to the negative log likelihood of the Laplace distribution


    Parameters
    ----------
    obs : array_like
        Observed values.
    location : array_like
        Location parameter of the forecast laplace distribution.
    scale : array_like
        Scale parameter of the forecast laplace distribution.
        The LS between obs and Laplace(location, scale).
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between obs and Laplace(location, scale).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_laplace(0.3, 0.1, 0.2)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale = B.asarray(scale)
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the laplace must be positive."
            )
    return logarithmic.laplace(obs, location, scale, backend=backend)


def logs_loglaplace(
    obs: "ArrayLike",
    /,
    locationlog: "ArrayLike" = 0.0,
    scalelog: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the log-Laplace distribution.

    This score is equivalent to the negative log likelihood of the log-Laplace distribution

    Parameters
    ----------
    obs : array_like
        Observed values.
    locationlog : array_like
        Location parameter of the forecast log-laplace distribution.
    scalelog : array_like
        Scale parameter of the forecast log-laplace distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between obs and Loglaplace(locationlog, scalelog).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_loglaplace(3.0, 0.1, 0.9)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scalelog = B.asarray(scalelog)
        if B.any(scalelog <= 0) or B.any(scalelog >= 1):
            raise ValueError(
                "`scalelog` contains entries outside of the range (0, 1). The scale parameter of the log-laplace distribution must be between 0 and 1."
            )
    return logarithmic.loglaplace(obs, locationlog, scalelog, backend=backend)


def logs_logistic(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the logistic distribution.

    This score is equivalent to the negative log likelihood of the logistic distribution

    Parameters
    ----------
    obs : array_like
        Observed values.
    mu : array_like
        Location parameter of the forecast logistic distribution.
    sigma : array_like
        Scale parameter of the forecast logistic distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS for the Logistic(mu, sigma) forecasts given the observations.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_logistic(0.0, 0.4, 0.1)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale = B.asarray(scale)
        if B.any(scale <= 0):
            raise ValueError(
                "`sigma` contains non-positive entries. The scale parameter of the logistic distribution must be positive."
            )
    return logarithmic.logistic(obs, location, scale, backend=backend)


def logs_loglogistic(
    obs: "ArrayLike",
    /,
    mulog: "ArrayLike" = 0.0,
    sigmalog: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the log-logistic distribution.

    This score is equivalent to the negative log likelihood of the log-logistic distribution

    Parameters
    ----------
    obs : array_like
        The observed values.
    mulog : array_like
        Location parameter of the log-logistic distribution.
    sigmalog : array_like
        Scale parameter of the log-logistic distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between obs and Loglogis(mulog, sigmalog).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_loglogistic(3.0, 0.1, 0.9)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        sigmalog = B.asarray(sigmalog)
        if B.any(sigmalog <= 0) or B.any(sigmalog >= 1):
            raise ValueError(
                "`sigmalog` contains entries outside of the range (0, 1). The scale parameter of the log-logistic distribution must be between 0 and 1."
            )
    return logarithmic.loglogistic(obs, mulog, sigmalog, backend=backend)


def logs_lognormal(
    obs: "ArrayLike",
    /,
    mulog: "ArrayLike" = 0.0,
    sigmalog: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the log-normal distribution.

    This score is equivalent to the negative log likelihood of the log-normal distribution

    Parameters
    ----------
    obs : array_like
        The observed values.
    mulog : array_like
        Mean of the normal underlying distribution.
    sigmalog : array_like
        Standard deviation of the underlying normal distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between Lognormal(mu, sigma) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_lognormal(0.0, 0.4, 0.1)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        sigmalog = B.asarray(sigmalog)
        if B.any(sigmalog <= 0):
            raise ValueError(
                "`sigmalog` contains non-positive entries. The scale parameter of the log-normal distribution must be positive."
            )
    return logarithmic.lognormal(obs, mulog, sigmalog, backend=backend)


def logs_mixnorm(
    obs: "ArrayLike",
    /,
    m: "ArrayLike" = 0.0,
    s: "ArrayLike" = 1.0,
    w: "ArrayLike" = None,
    mc_axis: "ArrayLike" = -1,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score for a mixture of normal distributions.

    This score is equivalent to the negative log likelihood of the normal mixture distribution

    Parameters
    ----------
    obs : array_like
        The observed values.
    m : array_like
        Means of the component normal distributions.
    s : array_like
        Standard deviations of the component normal distributions.
    w : array_like
        Non-negative weights assigned to each component.
    mc_axis : int
        The axis corresponding to the mixture components. Default is the last axis.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between MixNormal(m, s) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_mixnormal(0.0, [0.1, -0.3, 1.0], [0.4, 2.1, 0.7], [0.1, 0.2, 0.7])
    """
    B = backends.active if backend is None else backends[backend]
    obs, m, s = map(B.asarray, (obs, m, s))

    if w is None:
        M: int = m.shape[mc_axis]
        w = B.zeros(m.shape) + 1 / M
    else:
        w = B.asarray(w)
        w = w / B.sum(w, axis=mc_axis, keepdims=True)

    if check_pars:
        if B.any(s <= 0):
            raise ValueError(
                "`s` contains non-positive entries. The scale parameters of the normal distributions should be positive."
            )
        if B.any(w < 0):
            raise ValueError("`w` contains negative entries")

    if mc_axis != -1:
        m = B.moveaxis(m, mc_axis, -1)
        s = B.moveaxis(s, mc_axis, -1)
        w = B.moveaxis(w, mc_axis, -1)

    return logarithmic.mixnorm(obs, m, s, w, backend=backend)


def logs_negbinom(
    obs: "ArrayLike",
    /,
    n: "ArrayLike",
    prob: "ArrayLike | None" = None,
    *,
    mu: "ArrayLike | None" = None,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the negative binomial distribution.

    This score is equivalent to the negative log likelihood of the negative binomial distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    n : array_like
        Size parameter of the forecast negative binomial distribution.
    prob : array_like
        Probability parameter of the forecast negative binomial distribution.
    mu : array_like
        Mean of the forecast negative binomial distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between NegBinomial(n, prob) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_negbinom(2, 5, 0.5)

    Raises
    ------
    ValueError
        If both `prob` and `mu` are provided, or if neither is provided.
    """
    if (prob is None and mu is None) or (prob is not None and mu is not None):
        raise ValueError(
            "Either `prob` or `mu` must be provided, but not both or neither."
        )

    if prob is None:
        prob = n / (n + mu)

    if check_pars:
        B = backends.active if backend is None else backends[backend]
        prob = B.asarray(prob)
        if B.any(prob < 0) or B.any(prob > 1):
            raise ValueError("`prob` contains values outside the range [0, 1].")

    return logarithmic.negbinom(obs, n, prob, backend=backend)


def logs_normal(
    obs: "ArrayLike",
    /,
    mu: "ArrayLike" = 0.0,
    sigma: "ArrayLike" = 1.0,
    *,
    negative: bool = True,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "Array":
    r"""Compute the logarithmic score (LS) for the normal distribution.

    This score is equivalent to the (negative) log likelihood (if `negative = True`)

    Parameters
    ----------
    obs : array_like
        The observed values.
    mu : array_like
        Mean of the forecast normal distribution.
    sigma : array_like
        Standard deviation of the forecast normal distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between Normal(mu, sigma) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_normal(0.0, 0.4, 0.1)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        sigma = B.asarray(sigma)
        if B.any(sigma <= 0):
            raise ValueError(
                "`sigma` contains non-positive entries. The standard deviation of the normal distribution must be positive."
            )
    return logarithmic.normal(obs, mu, sigma, negative=negative, backend=backend)


def logs_2pnormal(
    obs: "ArrayLike",
    /,
    scale1: "ArrayLike" = 1.0,
    scale2: "ArrayLike" = 1.0,
    location: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the two-piece normal distribution.

    This score is equivalent to the negative log likelihood of the two-piece normal distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    scale1 : array_like
        Scale parameter of the lower half of the forecast two-piece normal distribution.
    scale2 : array_like
        Scale parameter of the upper half of the forecast two-piece normal distribution.
    location : array_like
        Location parameter of the forecast two-piece normal distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between 2pNormal(scale1, scale2, location) and obs.
    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_2pnormal(0.0, 0.4, 2.0, 0.1)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale1, scale2 = map(B.asarray, (scale1, scale2))
        if B.any(scale1 <= 0):
            raise ValueError(
                "`scale1` contains non-positive entries. The scale parameters of the two-piece normal distribution must be positive."
            )
        if B.any(scale2 <= 0):
            raise ValueError(
                "`scale2` contains non-positive entries. The scale parameters of the two-piece normal distribution must be positive."
            )
    return logarithmic.twopnormal(obs, scale1, scale2, location, backend=backend)


def logs_poisson(
    obs: "ArrayLike",
    /,
    mean: "ArrayLike",
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the Poisson distribution.

    This score is equivalent to the negative log likelihood of the Poisson distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    mean : array_like
        Mean parameter of the forecast poisson distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between Pois(mean) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_poisson(1, 2)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        mean = B.asarray(mean)
        if B.any(mean <= 0):
            raise ValueError(
                "`mean` contains non-positive entries. The mean parameter of the Poisson distribution must be positive."
            )
    return logarithmic.poisson(obs, mean, backend=backend)


def logs_t(
    obs: "ArrayLike",
    /,
    df: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the Student's t distribution.

    This score is equivalent to the negative log likelihood of the t distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    df : array_like
        Degrees of freedom parameter of the forecast t distribution.
    location : array_like
        Location parameter of the forecast t distribution.
    scale : array_like
        Scale parameter of the forecast t distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between t(df, location, scale) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_t(0.0, 0.1, 0.4, 0.1)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        df, scale = map(B.asarray, (df, scale))
        if B.any(df <= 0):
            raise ValueError(
                "`df` contains non-positive entries. The degrees of freedom parameter of the t distribution must be positive."
            )
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the t distribution must be positive."
            )
    return logarithmic.t(obs, df, location, scale, backend=backend)


def logs_tlogistic(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the truncated logistic distribution.

    This score is equivalent to the negative log likelihood of the truncated logistic distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    location : array_like
        Location parameter of the forecast distribution.
    scale : array_like
        Scale parameter of the forecast distribution.
    lower : array_like
        Lower boundary of the truncated forecast distribution.
    upper : array_like
        Upper boundary of the truncated forecast distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between tLogistic(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_tlogistic(0.0, 0.1, 0.4, -1.0, 1.0)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lower, upper = map(B.asarray, (scale, lower, upper))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the truncated logistic distribution must be positive."
            )
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    return logarithmic.tlogistic(obs, location, scale, lower, upper, backend=backend)


def logs_tnormal(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the truncated normal distribution.

    This score is equivalent to the negative log likelihood of the truncated normal distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    location : array_like
        Location parameter of the forecast distribution.
    scale : array_like
        Scale parameter of the forecast distribution.
    lower : array_like
        Lower boundary of the truncated forecast distribution.
    upper : array_like
        Upper boundary of the truncated forecast distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between tNormal(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_tnormal(0.0, 0.1, 0.4, -1.0, 1.0)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lower, upper = map(B.asarray, (scale, lower, upper))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the truncated normal distribution must be positive."
            )
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    return logarithmic.tnormal(obs, location, scale, lower, upper, backend=backend)


def logs_tt(
    obs: "ArrayLike",
    /,
    df: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the truncated Student's t distribution.

    This score is equivalent to the negative log likelihood of the truncated t distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    df : array_like
        Degrees of freedom parameter of the forecast distribution.
    location : array_like
        Location parameter of the forecast distribution.
    scale : array_like
        Scale parameter of the forecast distribution.
    lower : array_like
        Lower boundary of the truncated forecast distribution.
    upper : array_like
        Upper boundary of the truncated forecast distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between tt(df, location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_tt(0.0, 2.0, 0.1, 0.4, -1.0, 1.0)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lower, upper = map(B.asarray, (scale, lower, upper))
        if B.any(df <= 0):
            raise ValueError(
                "`df` contains non-positive entries. The degrees of freedom parameter of the truncated t distribution must be positive."
            )
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the truncated t distribution must be positive."
            )
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    return logarithmic.tt(obs, df, location, scale, lower, upper, backend=backend)


def logs_uniform(
    obs: "ArrayLike",
    /,
    min: "ArrayLike" = 0.0,
    max: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the uniform distribution.

    This score is equivalent to the negative log likelihood of the uniform distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    min : array_like
        Lower bound of the forecast uniform distribution.
    max : array_like
        Upper bound of the forecast uniform distribution.
    backend : str
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    score : array_like
        The LS between U(min, max) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_uniform(0.4, 0.0, 1.0)
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        min, max = map(B.asarray, (min, max))
        if B.any(min >= max):
            raise ValueError("`min` is not always smaller than `max`.")
    return logarithmic.uniform(obs, min, max, backend=backend)


__all__ = [
    "logs_beta",
    "logs_binomial",
    "logs_ensemble",
    "logs_exponential",
    "logs_exponential2",
    "logs_2pexponential",
    "logs_gamma",
    "logs_tlogistic",
    "logs_tnormal",
    "logs_tt",
    "logs_hypergeometric",
    "logs_logistic",
    "logs_loglogistic",
    "logs_lognormal",
    "logs_mixnorm",
    "logs_negbinom",
    "logs_normal",
    "logs_2pnormal",
    "logs_poisson",
    "logs_t",
    "logs_uniform",
    "clogs_ensemble",
]
