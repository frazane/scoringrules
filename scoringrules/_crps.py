import typing as tp

from scoringrules.backend import backends
from scoringrules.core import crps, stats

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def crps_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    m_axis: int = -1,
    *,
    ens_w: "Array" = None,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble.

    For a forecast :math:`F` and observation :math:`y`, the CRPS is formally defined as:

    .. math::
        \begin{align*}
            \mathrm{CRPS}(F, y) &= \int_{-\infty}^{\infty} (F(x)
            - \mathbf{1}\{y \le x\})^{2} dx \\
            &= \mathbb{E} | X - y | - \frac{1}{2} \mathbb{E} | X - X^{\prime} |,
        \end{align*}

    where :math:`X, X^{\prime} \sim F` are independent. When :math:`F` is the empirical
    distribution function of an ensemble forecast :math:`x_{1}, \dots, x_{M}`, the CRPS
    can be estimated in several ways. Currently, scoringrules supports several
    alternatives for ``estimator``:

    - the energy estimator (``"nrg"``),
    - the fair estimator (``"fair"``),
    - the probability weighted moment estimator (``"pwm"``),
    - the approximate kernel representation estimator (``"akr"``).
    - the AKR with circular permutation estimator (``"akr_circperm"``).
    - the integral estimator (``"int"``).
    - the quantile decomposition estimator (``"qd"``).

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like, shape (..., m)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    ens_w : array, shape (..., m)
        Weights assigned to the ensemble members. Array with the same shape as fct.
        Default is equal weighting.
    sorted_ensemble : bool
        Boolean indicating whether the ensemble members are already in ascending order.
        Default is False.
    estimator : str
        Indicates the CRPS estimator to be used.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.

    Returns
    -------
    crps : array_like
        The CRPS between the forecast ensemble and obs.

    See Also
    --------
    twcrps_ensemble, owcrps_ensemble, vrcrps_ensemble
        Weighted variants of the CRPS.
    crps_quantile
        CRPS for quantile forecasts.
    es_ensemble
        The multivariate equivalent of the CRPS.

    Notes
    -----
    :ref:`crps-estimators`
        Information on the different estimators available for the CRPS.

    References
    ----------
    .. [1] Matheson JE, Winkler RL (1976). “Scoring rules for continuous
        probability distributions.” Management Science, 22(10), 1087-1096.
        doi:10.1287/mnsc.22.10.1087.
    .. [2] Gneiting T, Raftery AE (2007). “Strictly proper scoring rules,
        prediction, and estimation.” Journal of the American Statistical
        Association, 102(477), 359-378. doi:10.1198/016214506000001437.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    >>> obs = rng.normal(size=3)
    >>> pred = rng.normal(size=(3, 10))
    >>> sr.crps_ensemble(obs, pred)
    array([0.69605316, 0.32865417, 0.39048665])
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))

    if m_axis != -1:
        fct = B.moveaxis(fct, m_axis, -1)

    sort_ensemble = not sorted_ensemble and estimator in ["qd", "pwm", "int"]

    if sort_ensemble:
        fct_uns = fct
        fct = B.sort(fct, axis=-1)

    if ens_w is None:
        if backend == "numba":
            if estimator not in crps.estimator_gufuncs:
                raise ValueError(
                    f"{estimator} is not a valid estimator. "
                    f"Must be one of {crps.estimator_gufuncs.keys()}"
                )
            return crps.estimator_gufuncs[estimator](obs, fct)

        return crps.ensemble(obs, fct, estimator, backend=backend)

    else:
        ens_w = B.asarray(ens_w)
        if B.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        ens_w = ens_w / B.sum(ens_w, axis=m_axis, keepdims=True)
        if m_axis != -1:
            ens_w = B.moveaxis(ens_w, m_axis, -1)
        if sort_ensemble:
            ind = B.argsort(fct_uns, axis=-1)
            ens_w = B.gather(ens_w, ind, axis=-1)

        if backend == "numba":
            if estimator not in crps.estimator_gufuncs:
                raise ValueError(
                    f"{estimator} is not a valid estimator. "
                    f"Must be one of {crps.estimator_gufuncs.keys()}"
                )
            return crps.estimator_gufuncs_w[estimator](obs, fct, ens_w)

        return crps.ensemble_w(obs, fct, ens_w, estimator, backend=backend)


def twcrps_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    ens_w: "Array" = None,
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"] = None,
    estimator: str = "pwm",
    sorted_ensemble: bool = False,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the threshold-weighted CRPS (twCRPS) for a finite ensemble.

    Computation is performed using the ensemble representation of the twCRPS in [1]_.

    .. math::
        \mathrm{twCRPS}(F_{ens}, y) = \frac{1}{M} \sum_{m = 1}^{M} |v(x_{m}) - v(y)|
          - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} |v(x_{m}) - v(x_{j})|,

    where :math:`F_{ens}(x) = \sum_{m=1}^{M} 1 \{ x_{m} \leq x \}/M` is the empirical
    distribution function associated with an ensemble forecast :math:`x_{1}, \dots, x_{M}` with
    :math:`M` members, and :math:`v` is the chaining function used to target particular outcomes.


    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    a : float
        The lower bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    b : float
        The upper bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    ens_w : array
        Weights assigned to the ensemble members. Array with the same shape as fct.
        Default is equal weighting.
    v_func : callable, array_like -> array_like
        Chaining function used to emphasise particular outcomes. For example, a function that
        only considers values above a certain threshold :math:`t` by projecting forecasts and observations
        to :math:`[t, \inf)`.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.

    Returns
    -------
    twcrps : array_like
        The twCRPS between the forecast ensemble and obs for the chosen chaining function.

    See Also
    --------
    crps_ensemble : Compute the CRPS for a finite ensemble.

    Notes
    -----
    :ref:`theory.weighted`
        Some theoretical background on weighted versions of scoring rules.

    References
    ----------
    .. [1] Allen, S., Ginsbourger, D., & Ziegel, J. (2023).
        Evaluating forecasts for high-impact events using transformed kernel scores.
        SIAM/ASA Journal on Uncertainty Quantification, 11(3), 906-940.
        Available at https://arxiv.org/abs/2202.12732.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    ...
    >>> def v_func(x):
    ...    return np.maximum(x, -1.0)
    ...
    >>> obs = rng.normal(size=3)
    >>> fct = rng.normal(size=(3, 10))
    >>> sr.twcrps_ensemble(obs, fct, v_func=v_func)
    array([0.69605316, 0.32865417, 0.39048665])
    """
    if v_func is None:
        B = backends.active if backend is None else backends[backend]
        a, b, obs, fct = map(B.asarray, (a, b, obs, fct))

        def v_func(x):
            return B.minimum(B.maximum(x, a), b)

    obs, fct = map(v_func, (obs, fct))
    return crps_ensemble(
        obs,
        fct,
        m_axis=m_axis,
        ens_w=ens_w,
        sorted_ensemble=sorted_ensemble,
        estimator=estimator,
        backend=backend,
    )


def owcrps_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    ens_w: "Array" = None,
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"] = None,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the outcome-weighted CRPS (owCRPS) for a finite ensemble.

    Computation is performed using the ensemble representation of the owCRPS in [1]_.

    .. math::
        \begin{aligned}
        \mathrm{owCRPS}(F_{ens}, y)
        &= \frac{1}{M \bar{w}} \sum_{m = 1}^{M} |x_{m} - y|\,w(x_{m})\,w(y)\\
        &\quad - \frac{1}{2 M^{2} \bar{w}^{2}}
        \sum_{m = 1}^{M} \sum_{j = 1}^{M} |x_{m} - x_{j}|\,
        w(x_{m})\,w(x_{j})\,w(y).
        \end{aligned}


    where :math:`F_{ens}(x) = \sum_{m=1}^{M} 1\{ x_{m} \leq x \}/M` is the empirical
    distribution function associated with an ensemble forecast
    :math:`x_{1}, \dots, x_{M}`  with :math:`M` members, :math:`w` is the chosen weight
    function, and :math:`\bar{w} = \sum_{m=1}^{M}w(x_{m})/M` is the average weight.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    a : float
        The lower bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    b : float
        The upper bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    ens_w : array
        Weights assigned to the ensemble members. Array with the same shape as fct.
        Default is equal weighting.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.

    Returns
    -------
    owcrps : array_like
        The owCRPS between the forecast ensemble and obs for the chosen weight function.

    Notes
    -----
    :ref:`theory.weighted`
        Some theoretical background on weighted versions of scoring rules.

    References
    ----------
    .. [1] Allen, S., Ginsbourger, D., & Ziegel, J. (2023).
        Evaluating forecasts for high-impact events using transformed kernel scores.
        SIAM/ASA Journal on Uncertainty Quantification, 11(3), 906-940.
        Available at https://arxiv.org/abs/2202.12732.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    ...
    >>> def w_func(x):
    ...    return (x > -1).astype(float)
    ...
    >>> obs = rng.normal(size=3)
    >>> fct = rng.normal(size=(3, 10))
    >>> sr.owcrps_ensemble(obs, fct, w_func=w_func)
    array([0.91103733, 0.45212402, 0.35686667])
    """

    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))

    if m_axis != -1:
        fct = B.moveaxis(fct, m_axis, -1)

    if w_func is None:

        def w_func(x):
            return ((a <= x) & (x <= b)) * 1.0

    obs_weights, fct_weights = map(w_func, (obs, fct))
    if B.any(obs_weights < 0) or B.any(fct_weights < 0):
        raise ValueError("`w_func` returns negative values")

    if ens_w is None:
        if backend == "numba":
            return crps.estimator_gufuncs["ownrg"](obs, fct, obs_weights, fct_weights)

        return crps.ow_ensemble(obs, fct, obs_weights, fct_weights, backend=backend)

    else:
        ens_w = B.asarray(ens_w)
        if B.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        ens_w = ens_w / B.sum(ens_w, axis=m_axis, keepdims=True)
        if m_axis != -1:
            ens_w = B.moveaxis(ens_w, m_axis, -1)

        if backend == "numba":
            return crps.estimator_gufuncs_w["ownrg"](
                obs, fct, obs_weights, fct_weights, ens_w
            )

        return crps.ow_ensemble_w(
            obs, fct, obs_weights, fct_weights, ens_w, backend=backend
        )


def vrcrps_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    ens_w: "Array" = None,
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"] = None,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the vertically re-scaled CRPS (vrCRPS) for a finite ensemble.

    Computation is performed using the ensemble representation of the vrCRPS in [1]_.

    .. math::
        \begin{split}
            \mathrm{vrCRPS}(F_{ens}, y) = & \frac{1}{M} \sum_{m = 1}^{M} |x_{m}
            - y|w(x_{m})w(y) - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} |x_{m}
            - x_{j}|w(x_{m})w(x_{j}) \\
            & + \left( \frac{1}{M} \sum_{m = 1}^{M} |x_{m}| w(x_{m})
            - |y| w(y) \right) \left( \frac{1}{M} \sum_{m = 1}^{M} w(x_{m}) - w(y) \right),
        \end{split}

    where :math:`F_{ens}(x) = \sum_{m=1}^{M} 1 \{ x_{m} \leq x \}/M` is the empirical
    distribution function associated with an ensemble forecast :math:`x_{1}, \dots, x_{M}` with
    :math:`M` members, :math:`w` is the chosen weight function, and :math:`\bar{w} = \sum_{m=1}^{M}w(x_{m})/M`.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    a : float
        The lower bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    b : float
        The upper bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    ens_w : array
        Weights assigned to the ensemble members. Array with the same shape as fct.
        Default is equal weighting.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.

    Returns
    -------
    vrcrps : array_like
        The vrCRPS between the forecast ensemble and obs for the chosen weight function.

    Notes
    -----
    :ref:`theory.weighted`
        Some theoretical background on weighted versions of scoring rules.

    References
    ----------
    .. [1] Allen, S., Ginsbourger, D., & Ziegel, J. (2023).
        Evaluating forecasts for high-impact events using transformed kernel scores.
        SIAM/ASA Journal on Uncertainty Quantification, 11(3), 906-940.
        Available at https://arxiv.org/abs/2202.12732.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    ...
    >>> def w_func(x):
    ...    return (x > -1).astype(float)
    ...
    >>> obs = rng.normal(size=3)
    >>> fct = rng.normal(size=(3, 10))
    >>> sr.vrcrps_ensemble(obs, fct, w_func)
    array([0.90036433, 0.41515255, 0.41653833])
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))

    if m_axis != -1:
        fct = B.moveaxis(fct, m_axis, -1)

    if w_func is None:

        def w_func(x):
            return ((a <= x) & (x <= b)) * 1.0

    obs_weights, fct_weights = map(w_func, (obs, fct))
    if B.any(obs_weights < 0) or B.any(fct_weights < 0):
        raise ValueError("`w_func` returns negative values")

    if ens_w is None:
        if backend == "numba":
            return crps.estimator_gufuncs["vrnrg"](obs, fct, obs_weights, fct_weights)

        return crps.vr_ensemble(obs, fct, obs_weights, fct_weights, backend=backend)

    else:
        ens_w = B.asarray(ens_w)
        if B.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        ens_w = ens_w / B.sum(ens_w, axis=m_axis, keepdims=True)
        if m_axis != -1:
            ens_w = B.moveaxis(ens_w, m_axis, -1)

        if backend == "numba":
            return crps.estimator_gufuncs_w["vrnrg"](
                obs, fct, obs_weights, fct_weights, ens_w
            )

        return crps.vr_ensemble_w(
            obs, fct, obs_weights, fct_weights, ens_w, backend=backend
        )


def crps_quantile(
    obs: "ArrayLike",
    fct: "Array",
    /,
    alpha: "Array",
    m_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Approximate the CRPS from quantile predictions via the Pinball Loss.

    It is based on the notation in [1]_.

    The CRPS can be approximated as the mean pinball loss for all
    quantile forecasts :math:`F_q` with level :math:`q \in Q`:

    .. math::
        \text{quantileCRPS} = \frac{2}{|Q|} \sum_{q \in Q} PB_q

    where the pinball loss is defined as:

    .. math::
        \text{PB}_q = \begin{cases}
            q(y - F_q) &\text{if} & y \geq F_q  \\
            (1-q)(F_q - y) &\text{else.} &  \\
        \end{cases}

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    alpha : array_like
        The percentile levels. We expect the quantile array to match the axis (see below) of the forecast array.
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.

    Returns
    -------
    qcrps : array_like
        An array of CRPS scores for each forecast, which should be averaged to get meaningful values.

    References
    ----------
    .. [1] Berrisch, J., & Ziel, F. (2023). CRPS learning.
        Journal of Econometrics, 237(2), 105221.
        Available at https://arxiv.org/abs/2102.00968.

    # TODO: add example, change reference
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct, alpha = map(B.asarray, (obs, fct, alpha))

    if B.any(alpha <= 0) or B.any(alpha >= 1):
        raise ValueError("`alpha` contains entries that are not between 0 and 1.")

    if m_axis != -1:
        fct = B.moveaxis(fct, m_axis, -1)

    if not fct.shape[-1] == alpha.shape[-1]:
        raise ValueError("Expected matching length of `fct` and `alpha` values.")

    if B.name == "numba":
        return crps.quantile_pinball_gufunc(obs, fct, alpha)

    return crps.quantile_pinball(obs, fct, alpha, backend=backend)


def crps_beta(
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
    r"""Compute the closed form of the CRPS for the beta distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \begin{split}
        \mathrm{CRPS}(F_{\alpha, \beta}, y) = & (u - l)\left\{ \frac{y - l}{u - l}
        \left( 2F_{\alpha, \beta} \left( \frac{y - l}{u - l} \right) - 1 \right) \right. \\
        & \left. + \frac{\alpha}{\alpha + \beta} \left( 1 - 2F_{\alpha + 1, \beta}
        \left( \frac{y - l}{u - l} \right)
        - \frac{2B(2\alpha, 2\beta)}{\alpha B(\alpha, \beta)^{2}} \right) \right\}
        \end{split}


    where :math:`F_{\alpha, \beta}` is the beta distribution function with
    shape parameters :math:`\alpha, \beta > 0`, and lower and upper bounds
    :math:`l, u \in \mathbb{R}`, :math:`l < u`.

    Parameters
    ----------
    obs : array_like
        The observed values.
    a: array_like
        First shape parameter of the forecast beta distribution.
    b: array_like
        Second shape parameter of the forecast beta distribution.
    lower : array_like
        Lower bound of the forecast beta distribution.
    upper : array_like
        Upper bound of the forecast beta distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between Beta(a, b) and obs.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_beta(0.3, 0.7, 1.1)
    0.08501024366637236
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
    return crps.beta(obs, a, b, lower, upper, backend=backend)


def crps_binomial(
    obs: "ArrayLike",
    /,
    n: "ArrayLike",
    prob: "ArrayLike",
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the binomial distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{n, p}, y) = 2 \sum_{x = 0}^{n} f_{n,p}(x) (1\{y < x\}
        - F_{n,p}(x) + f_{n,p}(x)/2) (x - y),

    where :math:`f_{n, p}` and :math:`F_{n, p}` are the PDF and CDF of the binomial distribution
    with size parameter :math:`n = 0, 1, 2, ...` and probability parameter :math:`p \in [0, 1]`.

    Parameters
    ----------
    obs : array_like
        The observed values as an integer or array of integers.
    n: array_like
        Size parameter of the forecast binomial distribution as an integer or array of integers.
    prob : array_like
        Probability parameter of the forecast binomial distribution as a float or array of floats.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps:
        The CRPS between Binomial(n, prob) and obs.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_binomial(4, 10, 0.5)
    0.5955772399902344
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        prob = B.asarray(prob)
        if B.any(prob < 0) or B.any(prob > 1):
            raise ValueError("`prob` contains values outside the range [0, 1].")
    return crps.binomial(obs, n, prob, backend=backend)


def crps_exponential(
    obs: "ArrayLike",
    /,
    rate: "ArrayLike",
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the exponential distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{\lambda}, y) = |y| - \frac{2F_{\lambda}(y)}{\lambda} + \frac{1}{2 \lambda},

    where :math:`F_{\lambda}` is exponential distribution function with rate parameter :math:`\lambda > 0`.

    Parameters
    ----------
    obs : array_like
        The observed values.
    rate : array_like
        Rate parameter of the forecast exponential distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps:
        The CRPS between Exp(rate) and obs.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> import numpy as np
    >>> sr.crps_exponential(0.8, 3.0)
    0.360478635526275
    >>> sr.crps_exponential(np.array([0.8, 0.9]), np.array([3.0, 2.0]))
    array([0.36047864, 0.31529889])
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        rate = B.asarray(rate)
        if B.any(rate <= 0):
            raise ValueError(
                "`rate` contains non-positive entries. The rate parameter of the exponential distribution must be positive."
            )
    return crps.exponential(obs, rate, backend=backend)


def crps_exponentialM(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    mass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the standard exponential distribution with a point mass at the boundary.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{M}, y) = |y| - 2 (1 - M) F(y) + \frac{(1 - M)**2}{2},

    .. math::
        \mathrm{CRPS}(F_{M, \mu, \sigma}, y) =
        \sigma \mathrm{CRPS} \left( F_{M}, \frac{y - \mu}{\sigma} \right),

    where :math:`F_{M, \mu, \sigma}` is standard exponential distribution function
    generalised using a location parameter :math:`\mu` and scale parameter :math:`\sigma < 0`
    and a point mass :math:`M \in [0, 1]` at :math:`\mu`, :math:`F_{M} = F_{M, 0, 1}`, and

    .. math::
        F(y) = 1 - \exp(-y)

    for :math:`y \geq 0`, and 0 otherwise.

    Parameters
    ----------
    obs : array_like
        The observed values.
    mass : array_like
        Mass parameter of the forecast exponential distribution.
    location : array_like
        Location parameter of the forecast exponential distribution.
    scale : array_like
        Scale parameter of the forecast exponential distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between obs and ExpM(mass, location, scale).

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_exponentialM(0.4, 0.2, 0.0, 1.0)
    0.19251207365702294
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, mass = map(B.asarray, (scale, mass))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter must be positive."
            )
        if B.any(mass < 0) or B.any(mass > 1):
            raise ValueError("`mass` contains entries outside the range [0, 1].")
    return crps.exponentialM(obs, location, scale, mass, backend=backend)


def crps_2pexponential(
    obs: "ArrayLike",
    /,
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the two-piece exponential distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{\sigma_{1}, \sigma_{2}, \mu}, y) = |y - \mu| +
        \frac{2\sigma_{\pm}^{2}}{\sigma_{1} + \sigma_{2}} \exp \left( - \frac{|y - \mu|}{\sigma_{\pm}} \right) -
        \frac{2\sigma_{\pm}^{2}}{\sigma_{1} + \sigma_{2}} + \frac{\sigma_{1}^{3} + \sigma_{2}^{3}}{2(\sigma_{1} + \sigma_{2})^2},

    where :math:`F_{\sigma_{1}, \sigma_{2}, \mu}` is the two-piece exponential distribution function
    with scale parameters :math:`\sigma_{1}, \sigma_{2} > 0` and location parameter :math:`\mu`. The
    parameter :math:`\sigma_{\pm}` is equal to :math:`\sigma_{1}` if :math:`y < 0` and :math:`\sigma_{2}` if :math:`y \geq 0`.

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
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between 2pExp(sigma1, sigma2, location) and obs.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_2pexponential(0.8, 3.0, 1.4, 0.0)
    array(1.18038524)
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
    return crps.twopexponential(obs, scale1, scale2, location, backend=backend)


def crps_gamma(
    obs: "ArrayLike",
    /,
    shape: "ArrayLike",
    rate: "ArrayLike | None" = None,
    *,
    scale: "ArrayLike | None" = None,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the gamma distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{\alpha, \beta}, y) = y(2F_{\alpha, \beta}(y) - 1)
        - \frac{\alpha}{\beta} (2 F_{\alpha + 1, \beta}(y) - 1)
        - \frac{1}{\beta B(1/2, \alpha)},

    where :math:`F_{\alpha, \beta}` is gamma distribution function with shape
    parameter :math:`\alpha > 0` and rate parameter :math:`\beta > 0` (equivalently,
    with scale parameter :math:`1/\beta`).

    Parameters
    ----------
    obs : array_like
        The observed values.
    shape: array_like
        Shape parameter of the forecast gamma distribution.
    rate : array_like, optional
        Rate parameter of the forecast rate distribution.
        Either ``rate`` or ``scale`` must be provided.
    scale : array_like, optional
        Scale parameter of the forecast scale distribution, where ``scale = 1 / rate``.
        Either ``rate`` or ``scale`` must be provided.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps:
        The CRPS between obs and Gamma(shape, rate).

    References
    ----------
    .. [1] Michael Scheuerer. David Möller. "Probabilistic wind speed forecasting
        on a grid based on ensemble model output statistics."
        Ann. Appl. Stat. 9 (3) 1328 - 1349, September 2015.
        https://doi.org/10.1214/15-AOAS843

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gamma(0.2, 1.1, 0.1)
    5.503536008961291

    Raises
    ------
    ValueError
        If both ``rate`` and ``scale`` are provided, or if neither is provided.
    """
    if (scale is None and rate is None) or (scale is not None and rate is not None):
        raise ValueError(
            "Either ``rate`` or ``scale`` must be provided, but not both or neither."
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

    return crps.gamma(obs, shape, rate, backend=backend)


def crps_gev(
    obs: "ArrayLike",
    /,
    shape: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the generalised extreme value (GEV) distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \text{CRPS}(F_{\xi, \mu, \sigma}, y) =
        \sigma \cdot \text{CRPS}(F_{\xi}, \frac{y - \mu}{\sigma})

    see Notes below for special cases.

    Parameters
    ----------
    obs : array_like
        The observed values.
    shape : array_like
        Shape parameter of the forecast GEV distribution.
    location : array_like, optional
        Location parameter of the forecast GEV distribution.
    scale : array_like, optional
        Scale parameter of the forecast GEV distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between obs and GEV(shape, location, scale).

    Notes
    -----

    Special cases are handled as follows:

    - For :math:`\xi = 0`:

    .. math::
        \text{CRPS}(F_{\xi}, y) = -y - 2\text{Ei}(\log F_{\xi}(y)) + C - \log 2

    - For :math:`\xi \neq 0`:

    .. math::
        \text{CRPS}(F_{\xi}, y) = y(2F_{\xi}(y) - 1) - 2G_{\xi}(y)
        - \frac{1 - (2 - 2^{\xi}) \Gamma(1 - \xi)}{\xi}

    where :math:`C` is the Euler-Mascheroni constant, :math:`\text{Ei}` is the exponential
    integral, and :math:`\Gamma` is the gamma function. The GEV cumulative distribution
    function :math:`F_{\xi}` and the auxiliary function :math:`G_{\xi}` are defined as:

    - For :math:`\xi = 0`:

    .. math::
        F_{\xi}(x) = \exp(-\exp(-x))

    - For :math:`\xi \neq 0`:

    .. math::
        F_{\xi}(x) =
        \begin{cases}
        0, & x \leq \frac{1}{\xi} \\
        \exp(-(1 + \xi x)^{-1/\xi}), & x > \frac{1}{\xi}
        \end{cases}

    .. math::
        G_{\xi}(x) =
        \begin{cases}
        0, & x \leq \frac{1}{\xi} \\
        \frac{F_{\xi}(x)}{\xi} + \frac{\Gamma_u(1-\xi, -\log F_{\xi}(x))}{\xi}, & x > \frac{1}{\xi}
        \end{cases}

    References
    ----------
    .. [1] Friederichs, P., & Thorarinsdottir, T. L. (2012).
        A comparison of parametric and non-parametric methods for
        forecasting extreme events. Environmetrics, 23(7), 595-611.
        https://doi.org/10.1002/env.2176


    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gev(0.3, 0.1)
    0.2924712413052034
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, shape = map(B.asarray, (scale, shape))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the GEV distribution must be positive."
            )
        if B.any(shape >= 1):
            raise ValueError(
                "`shape` contains entries larger than 1. The CRPS for the GEV distribution is only valid for shape values less than 1."
            )
    return crps.gev(obs, shape, location, scale, backend=backend)


def crps_gpd(
    obs: "ArrayLike",
    /,
    shape: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    mass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the generalised pareto distribution (GPD).

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{M, \xi}, y) =
        |y| - \frac{2 (1 - M)}{1 - \xi} \left( 1 - (1 - F_{\xi}(y))^{1 - \xi} \right)
        + \frac{(1 - M)^{2}}{2 - \xi},

    .. math::
        \mathrm{CRPS}(F_{M, \xi, \mu, \sigma}, y) =
        \sigma \mathrm{CRPS} \left( F_{M, \xi}, \frac{y - \mu}{\sigma} \right),

    where :math:`F_{M, \xi, \mu, \sigma}` is the GPD distribution function with shape
    parameter :math:`\xi < 1`, location parameter :math:`\mu`, scale parameter :math:`\sigma > 0`,
    and point mass :math:`M \in [0, 1]` at the lower boundary. :math:`F_{M, \xi} = F_{M, \xi, 0, 1}`.

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
    mass : array_like
        Mass parameter at the lower boundary of the forecast GPD distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between obs and GPD(shape, location, scale, mass).

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gpd(0.3, 0.9)
    0.6849331901197213
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, shape, mass = map(B.asarray, (scale, shape, mass))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the GPD distribution must be positive. `nan` is returned in these places."
            )
        if B.any(shape >= 1):
            raise ValueError(
                "`shape` contains entries larger than 1. The CRPS for the GPD distribution is only valid for shape values less than 1. `nan` is returned in these places."
            )
        if B.any(mass < 0) or B.any(mass > 1):
            raise ValueError(
                "`mass` contains entries outside the range [0, 1]. `nan` is returned in these places."
            )
    return crps.gpd(obs, shape, location, scale, mass, backend=backend)


def crps_gtclogistic(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    lmass: "ArrayLike" = 0.0,
    umass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the generalised truncated and censored logistic distribution.

    .. math::
        \begin{aligned}
        \mathrm{CRPS}(F_{l, L}^{u, U}, y) = & \, |y - z| + uU^{2} - lL^{2} \\
        & - \left( \frac{1 - L - U}{F(u) - F(l)} \right) z
        \left( \frac{(1 - 2L) F(u) + (1 - 2U) F(l)}{1 - L - U} \right) \\
        & - \left( \frac{1 - L - U}{F(u) - F(l)} \right)
        \left( 2 \log F(-z) - 2G(u)U - 2 G(l)L \right) \\
        & - \left( \frac{1 - L - U}{F(u) - F(l)} \right)^{2} \left( H(u) - H(l) \right),
        \end{aligned}

    .. math::
        \begin{aligned}
        \mathrm{CRPS}(F_{l, L, \mu, \sigma}^{u, U}, y) = & \, \sigma
        \mathrm{CRPS}(F_{(l - \mu)/\sigma, L}^{(u - \mu)/\sigma, U}, \frac{y - \mu}{\sigma}),
        \end{aligned}

    .. math::
        G(x) = xF(x) + \log F(-x),

    ..math::
        H(x) = F(x) - xF(x)^{2} + (1 - 2F(x))\log F(-x),

    where :math:`F` is the CDF of the standard logistic distribution, :math:`F_{l, L, \mu, \sigma}^{u, U}`
    is the CDF of the logistic distribution truncated below at :math:`l` and above at :math:`u`,
    with point masses :math:`L, U > 0` at the lower and upper boundaries, respectively, and
    location and scale parameters :math:`\mu` and :math:`\sigma > 0`.

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
    lmass : array_like
        Point mass assigned to the lower boundary of the forecast distribution.
    umass : array_like
        Point mass assigned to the upper boundary of the forecast distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between gtcLogistic(location, scale, lower, upper, lmass, umass) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gtclogistic(0.0, 0.1, 0.4, -1.0, 1.0, 0.1, 0.1)
    0.1658713056903939
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lmass, umass, lower, upper = map(
            B.asarray, (scale, lmass, umass, lower, upper)
        )
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the generalised logistic distribution must be positive."
            )
        if B.any(lmass < 0) or B.any(lmass > 1):
            raise ValueError("`lmass` contains entries outside the range [0, 1].")
        if B.any(umass < 0) or B.any(umass > 1):
            raise ValueError("`umass` contains entries outside the range [0, 1].")
        if B.any(umass + lmass >= 1):
            raise ValueError("The sum of `umass` and `lmass` should be smaller than 1.")
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    return crps.gtclogistic(
        obs,
        location,
        scale,
        lower,
        upper,
        lmass,
        umass,
        backend=backend,
    )


def crps_tlogistic(
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
    r"""Compute the closed form of the CRPS for the truncated logistic distribution.

    It is based on the formulation for the generalised truncated and censored logistic
    distribution with lmass and umass set to zero.

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
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between tLogistic(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_tlogistic(0.0, 0.1, 0.4, -1.0, 1.0)
    0.12714830546327846
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
    return crps.gtclogistic(
        obs,
        location,
        scale,
        lower,
        upper,
        0.0,
        0.0,
        backend=backend,
    )


def crps_clogistic(
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
    r"""Compute the closed form of the CRPS for the censored logistic distribution.

    It is based on the formulation for the generalised truncated and censored logistic distribution with
    lmass and umass set to the tail probabilities of the predictive distribution.

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
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between cLogistic(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_clogistic(0.0, 0.1, 0.4, -1.0, 1.0)
    0.15805632276434345
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lower, upper = map(B.asarray, (scale, lower, upper))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the censored logistic distribution must be positive."
            )
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    lmass = stats._logis_cdf((lower - location) / scale)
    umass = 1 - stats._logis_cdf((upper - location) / scale)
    return crps.gtclogistic(
        obs,
        location,
        scale,
        lower,
        upper,
        lmass,
        umass,
        backend=backend,
    )


def crps_gtcnormal(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    lmass: "ArrayLike" = 0.0,
    umass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the generalised truncated and censored normal distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{l, L}^{u, U}, y) = |y - z| + uU^{2} - lL^{2}
        + \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right) z \left( 2 \Phi(z) - \frac{(1 - 2L) \Phi(u) + (1 - 2U) \Phi(l)}{1 - L - U} \right)
        + \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right) \left( 2 \phi(z) - 2 \phi(u)U - 2 \phi(l)L \right)
        - \left( \frac{1 - L - U}{\Phi(u) - \Phi(l)} \right)^{2} \left( \frac{1}{\sqrt{\pi}} \right) \left( \Phi(u \sqrt{2}) - \Phi(l \sqrt{2}) \right),

    .. math::
        \mathrm{CRPS}(F_{l, L, \mu, \sigma}^{u, U}, y) = \sigma \mathrm{CRPS}(F_{(l - \mu)/\sigma, L}^{(u - \mu)/\sigma, U}, \frac{y - \mu}{\sigma}),

    where :math:`\Phi:math:` and :math:`\phi` are respectively the CDF and PDF of the standard normal
    distribution, :math:`F_{l, L, \mu, \sigma}^{u, U}` is the CDF of the normal distribution
    truncated below at :math:`l` and above at :math:`u`, with point masses :math:`L, U > 0` at the
    lower and upper boundaries, respectively, and location and scale parameters :math:`\mu` and :math:`\sigma > 0`.
    :math:`F_{l, L}^{u, U} = F_{l, L, 0, 1}^{u, U}`.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gtcnormal(0.0, 0.1, 0.4, -1.0, 1.0, 0.1, 0.1)
    0.1351100832878575
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lmass, umass, lower, upper = map(
            B.asarray, (scale, lmass, umass, lower, upper)
        )
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the generalised normal distribution must be positive."
            )
        if B.any(lmass < 0) or B.any(lmass > 1):
            raise ValueError("`lmass` contains entries outside the range [0, 1].")
        if B.any(umass < 0) or B.any(umass > 1):
            raise ValueError("`umass` contains entries outside the range [0, 1].")
        if B.any(umass + lmass >= 1):
            raise ValueError("The sum of `umass` and `lmass` should be smaller than 1.")
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    return crps.gtcnormal(
        obs,
        location,
        scale,
        lower,
        upper,
        lmass,
        umass,
        backend=backend,
    )


def crps_tnormal(
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
    r"""Compute the closed form of the CRPS for the truncated normal distribution.

    It is based on the formulation for the generalised truncated and censored normal distribution with
    distribution with lmass and umass set to zero.

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
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between tNormal(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_tnormal(0.0, 0.1, 0.4, -1.0, 1.0)
    0.10070146718008832
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
    return crps.gtcnormal(obs, location, scale, lower, upper, 0.0, 0.0, backend=backend)


def crps_cnormal(
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
    r"""Compute the closed form of the CRPS for the censored normal distribution.

    It is based on the formulation for the generalised truncated and censored normal distribution with
    lmass and umass set to the tail probabilities of the predictive distribution.

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
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between cNormal(location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_cnormal(0.0, 0.1, 0.4, -1.0, 1.0)
    0.10338851213123085
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lower, upper = map(B.asarray, (scale, lower, upper))
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the censored normal distribution must be positive."
            )
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    lmass = stats._norm_cdf((lower - location) / scale)
    umass = 1 - stats._norm_cdf((upper - location) / scale)
    return crps.gtcnormal(
        obs,
        location,
        scale,
        lower,
        upper,
        lmass,
        umass,
        backend=backend,
    )


def crps_gtct(
    obs: "ArrayLike",
    /,
    df: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    lower: "ArrayLike" = float("-inf"),
    upper: "ArrayLike" = float("inf"),
    lmass: "ArrayLike" = 0.0,
    umass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the generalised truncated and censored t distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{l, L, \nu}^{u, U}, y) = |y - z| + uU^{2} - lL^{2} + \left( \frac{1 - L - U}{F_{\nu}(u) - F_{\nu}(l)} \right) z \left( 2 F_{\nu}(z) - \frac{(1 - 2L) F_{\nu}(u) + (1 - 2U) F_{\nu}(l)}{1 - L - U} \right) - \left( \frac{1 - L - U}{F_{\nu}(u) - F_{\nu}(l)} \right) \left( 2 G_{\nu}(z) - 2 G_{\nu}(u)U - 2 G_{\nu}(l)L \right) - \left( \frac{1 - L - U}{F_{\nu}(u) - F_{\nu}(l)} \right)^{2} \bar{B}_{\nu} \left( H_{\nu}(u) - H_{\nu}(l) \right),

    .. math::
        \mathrm{CRPS}(F_{l, L, \nu, \mu, \sigma}^{u, U}, y) = \sigma \mathrm{CRPS}(F_{(l - \mu)/\sigma, L, \nu}^{(u - \mu)/\sigma, U}, \frac{y - \mu}{\sigma}),

    .. math::
        G_{\nu}(x) = - \left( \frac{\nu + x^{2}}{\nu - 1} \right) f_{\nu}(x),

    .. math::
        H_{\nu}(x) = \frac{1}{2} + \frac{1}{2} \mathrm{sgn}(x) I \left( \frac{1}{2}, \nu - \frac{1}{2}, \frac{x^{2}}{\nu + x^{2}} \right),

    .. math::
        \bar{B}_{\nu} = \left( \frac{2 \sqrt{\nu}}{\nu - 1} \right) \frac{B(\frac{1}{2}, \nu - \frac{1}{2})}{B(\frac{1}{2}, \frac{\nu}{2})^{2}},

    where :math:`F_{\nu}` is the CDF of the standard t distribution with :math:`\nu > 1` degrees of freedom,
    distribution, :math:`F_{l, L, \nu, \mu, \sigma}^{u, U}` is the CDF of the t distribution
    truncated below at :math:`l` and above at :math:`u`, with point masses :math:`L, U > 0` at the lower and upper
    boundaries, respectively, and degrees of freedom, location and scale parameters :math:`\nu > 1`, :math:`\mu` and :math:`\sigma > 0`.
    :math:`F_{l, L, \nu}^{u, U} = F_{l, L, \nu, 0, 1}^{u, U}`.

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
    lmass : array_like
        Point mass assigned to the lower boundary of the forecast distribution.
    umass : array_like
        Point mass assigned to the upper boundary of the forecast distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between gtct(df, location, scale, lower, upper, lmass, umass) and obs.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_gtct(0.0, 2.0, 0.1, 0.4, -1.0, 1.0, 0.1, 0.1)
    0.13997789333289662
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lmass, umass, lower, upper = map(
            B.asarray, (scale, lmass, umass, lower, upper)
        )
        if B.any(df <= 0):
            raise ValueError(
                "`df` contains non-positive entries. The degrees of freedom parameter of the generalised t distribution must be positive."
            )
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the generalised t distribution must be positive."
            )
        if B.any(lmass < 0) or B.any(lmass > 1):
            raise ValueError("`lmass` contains entries outside the range [0, 1].")
        if B.any(umass < 0) or B.any(umass > 1):
            raise ValueError("`umass` contains entries outside the range [0, 1].")
        if B.any(umass + lmass >= 1):
            raise ValueError("The sum of `umass` and `lmass` should be smaller than 1.")
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    return crps.gtct(
        obs,
        df,
        location,
        scale,
        lower,
        upper,
        lmass,
        umass,
        backend=backend,
    )


def crps_tt(
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
    r"""Compute the closed form of the CRPS for the truncated t distribution.

    It is based on the formulation for the generalised truncated and censored t distribution with
    lmass and umass set to zero.

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
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between tt(df, location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_tt(0.0, 2.0, 0.1, 0.4, -1.0, 1.0)
    0.10323007471747117
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
    return crps.gtct(
        obs,
        df,
        location,
        scale,
        lower,
        upper,
        0.0,
        0.0,
        backend=backend,
    )


def crps_ct(
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
    r"""Compute the closed form of the CRPS for the censored t distribution.

    It is based on the formulation for the generalised truncated and censored t distribution with
    lmass and umass set to the tail probabilities of the predictive distribution.

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
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between ct(df, location, scale, lower, upper) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_ct(0.0, 2.0, 0.1, 0.4, -1.0, 1.0)
    0.12672580744453948
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale, lower, upper = map(B.asarray, (scale, lower, upper))
        if B.any(df <= 0):
            raise ValueError(
                "`df` contains non-positive entries. The degrees of freedom parameter of the censored t distribution must be positive."
            )
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the censored t distribution must be positive."
            )
        if B.any(lower >= upper):
            raise ValueError("`lower` is not always smaller than `upper`.")
    lmass = stats._t_cdf((lower - location) / scale, df)
    umass = 1 - stats._t_cdf((upper - location) / scale, df)
    return crps.gtct(
        obs,
        df,
        location,
        scale,
        lower,
        upper,
        lmass,
        umass,
        backend=backend,
    )


def crps_hypergeometric(
    obs: "ArrayLike",
    /,
    m: "ArrayLike",
    n: "ArrayLike",
    k: "ArrayLike",
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the hypergeometric distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{m, n, k}, y) = 2 \sum_{x = 0}^{n} f_{m,n,k}(x) (1\{y < x\}
        - F_{m,n,k}(x) + f_{m,n,k}(x)/2) (x - y),

    where :math:`f_{m, n, k}` and :math:`F_{m, n, k}` are the PDF and CDF of the hypergeometric
    distribution with population parameters :math:`m,n = 0, 1, 2, ...` and size parameter
    :math:`k = 0, ..., m + n`.

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
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps:
        The CRPS between obs and Hypergeometric(m, n, k).

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_hypergeometric(5, 7, 13, 12)
    0.44697415547610597
    """
    # TODO: add check that m,n,k are integers
    return crps.hypergeometric(obs, m, n, k, backend=backend)


def crps_laplace(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the laplace distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F, y) = |y - \mu|
        + \sigma \exp ( -| y - \mu| / \sigma) - \frac{3\sigma}{4},

    where :math:`\mu` and :math:`\sigma > 0` are the location and scale parameters
    of the Laplace distribution.

    Parameters
    ----------
    obs : array_like
        Observed values.
    location : array_like
        Location parameter of the forecast laplace distribution.
    scale : array_like
        Scale parameter of the forecast laplace distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps:
        The CRPS between obs and Laplace(location, scale).

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_laplace(0.3, 0.1, 0.2)
    0.12357588823428847
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale = B.asarray(scale)
        if B.any(scale <= 0):
            raise ValueError(
                "`scale` contains non-positive entries. The scale parameter of the laplace must be positive."
            )
    return crps.laplace(obs, location, scale, backend=backend)


def crps_logistic(
    obs: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the logistic distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(\mathcal{L}(\mu, \sigma), y) = \sigma \left\{ \omega - 2 \log F(\omega) - 1 \right\},

    where :math:`F(\omega)` is the CDF of the standard logistic distribution at the
    normalized prediction error :math:`\omega = \frac{y - \mu}{\sigma}`.

    Parameters
    ----------
    obs : array_like
        Observed values.
    location: array_like
        Location parameter of the forecast logistic distribution.
    scale: array_like
        Scale parameter of the forecast logistic distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS for the Logistic(location, scale) forecasts given the observations.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_logistic(0.0, 0.4, 0.1)
    0.3036299855835619
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scale = B.asarray(scale)
        if B.any(scale <= 0):
            raise ValueError(
                "`sigma` contains non-positive entries. The scale parameter of the logistic distribution must be positive."
            )
    return crps.logistic(obs, location, scale, backend=backend)


def crps_loglaplace(
    obs: "ArrayLike",
    /,
    locationlog: "ArrayLike" = 0.0,
    scalelog: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the log-Laplace distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{\mu, \sigma}, y) = y (2 F_{\mu, \sigma}(y) - 1)
        + \exp(\mu) \left( \frac{\sigma}{4 - \sigma^{2}} + A(y) \right),

    where :math:`F_{\mu, \sigma}` is the CDF of the log-laplace distribution with location
    parameter :math:`\mu` and scale parameter :math:`\sigma \in (0, 1)`, and

    .. math::
        A(y) = \frac{1}{1 + \sigma} \left( 1 - (2 F_{\mu, \sigma}(y) - 1)^{1 + \sigma} \right),

    if :math:`y < \exp{\mu}`, and

    .. math::
        A(y) = \frac{-1}{1 - \sigma} \left( 1 - (2 (1 - F_{\mu, \sigma}(y)))^{1 - \sigma} \right),

    if :math:`y \ge \exp{\mu}`.

    Parameters
    ----------
    obs : array_like
        Observed values.
    locationlog : array_like
        Location parameter of the forecast log-laplace distribution.
    scalelog : array_like
        Scale parameter of the forecast log-laplace distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between obs and Loglaplace(locationlog, scalelog).

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_loglaplace(3.0, 0.1, 0.9)
    1.162020513653791
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        scalelog = B.asarray(scalelog)
        if B.any(scalelog <= 0) or B.any(scalelog >= 1):
            raise ValueError(
                "`scalelog` contains entries outside of the range (0, 1). The scale parameter of the log-laplace distribution must be between 0 and 1."
            )
    return crps.loglaplace(obs, locationlog, scalelog, backend=backend)


def crps_loglogistic(
    obs: "ArrayLike",
    /,
    mulog: "ArrayLike" = 0.0,
    sigmalog: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the log-logistic distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \text{CRPS}(y, F_{\mu,\sigma}) = y \left( 2F_{\mu,\sigma}(y) - 1 \right)
        - 2 \exp(\mu) I\left(F_{\mu,\sigma}(y); 1 + \sigma, 1 - \sigma\right) \\
        + \exp(\mu)(1 - \sigma) B(1 + \sigma, 1 - \sigma)

    where \( F_{\mu,\sigma}(x) \) is the cumulative distribution function (CDF) of
    the log-logistic distribution, defined as:

    .. math::
        F_{\mu,\sigma}(x) =
        \begin{cases}
        0, & x \leq 0 \\
        \left( 1 + \exp\left(-\frac{\log x - \mu}{\sigma}\right) \right)^{-1}, & x > 0
        \end{cases}

    :math:`B` is the beta function, and :math:`I` is the regularised incomplete beta function.

    Parameters
    ----------
    obs : array_like
        The observed values.
    mulog : array_like
        Location parameter of the log-logistic distribution.
    sigmalog : array_like
        Scale parameter of the log-logistic distribution.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between obs and Loglogis(mulog, sigmalog).

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_loglogistic(3.0, 0.1, 0.9)
    1.1329527730161177
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        sigmalog = B.asarray(sigmalog)
        if B.any(sigmalog <= 0) or B.any(sigmalog >= 1):
            raise ValueError(
                "`sigmalog` contains entries outside of the range (0, 1). The scale parameter of the log-logistic distribution must be between 0 and 1."
            )
    return crps.loglogistic(obs, mulog, sigmalog, backend=backend)


def crps_lognormal(
    obs: "ArrayLike",
    /,
    mulog: "ArrayLike" = 0.0,
    sigmalog: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the lognormal distribution.

    It is based on the formulation introduced by [1]_:

    .. math::
        \mathrm{CRPS}(\mathrm{log}\mathcal{N}(\mu, \sigma), y) =
        y [2 \Phi(y) - 1] - 2 \mathrm{exp}(\mu + \frac{\sigma^2}{2})
        \left[ \Phi(\omega - \sigma) + \Phi(\frac{\sigma}{\sqrt{2}}) \right],

    where :math:`\Phi` is the CDF of the standard normal distribution and
    :math:`\omega = \frac{\mathrm{log}y - \mu}{\sigma}`.

    Note that mean and standard deviation are not the values for the distribution itself,
    but of the underlying normal distribution it is derived from.

    Parameters
    ----------
    obs : array_like
        The observed values.
    mulog : array_like
        Mean of the normal underlying distribution.
    sigmalog : array_like
        Standard deviation of the underlying normal distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between Lognormal(mu, sigma) and obs.

    References
    ----------
    .. [1] Baran, S. and Lerch, S. (2015), Log-normal distribution based
        Ensemble Model Output Statistics models for probabilistic wind-speed forecasting.
        Q.J.R. Meteorol. Soc., 141: 2289-2299. https://doi.org/10.1002/qj.2521

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_lognormal(0.1, 0.4, 0.0)
    1.3918246976412703
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        sigmalog = B.asarray(sigmalog)
        if B.any(sigmalog <= 0):
            raise ValueError(
                "`sigmalog` contains non-positive entries. The scale parameter of the log-normal distribution must be positive."
            )
    return crps.lognormal(obs, mulog, sigmalog, backend=backend)


def crps_mixnorm(
    obs: "ArrayLike",
    /,
    m: "ArrayLike" = 0.0,
    s: "ArrayLike" = 1.0,
    w: "ArrayLike" = None,
    m_axis: "ArrayLike" = -1,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for a mixture of normal distributions.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F, y) = \sum_{i=1}^{M} w_{i} A(y - \mu_{i}, \sigma_{i}^{2}) - \frac{1}{2} \sum_{i=1}^{M} \sum_{j=1}^{M} w_{i} w_{j} A(\mu_{i} - \mu_{j}, \sigma_{i}^{2} + \sigma_{j}^{2}),

    where :math:`F(x) = \sum_{i=1}^{M} w_{i} \Phi \left( \frac{x - \mu_{i}}{\sigma_{i}} \right)`,
    and :math:`A(\mu, \sigma^{2}) = \mu (2 \Phi(\frac{\mu}{\sigma}) - 1) + 2\sigma \phi(\frac{\mu}{\sigma}).`

    Parameters
    ----------
    obs : array_like
        The observed values.
    m: array_like
        Means of the component normal distributions.
    s: array_like
        Standard deviations of the component normal distributions.
    w: array_like
        Non-negative weights assigned to each component.
    m_axis : int
        The axis corresponding to the mixture components. Default is the last axis.
    backend : str, optional
        The name of the backend used for computations. Defaults to ``numba`` if available, else ``numpy``.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between MixNormal(m, s) and obs.

    References
    ----------
    .. [1] Grimit, E.P., Gneiting, T., Berrocal, V.J. and Johnson, N.A. (2006),
        The continuous ranked probability score for circular variables and its
        application to mesoscale forecast ensemble verification.
        Q.J.R. Meteorol. Soc., 132: 2925-2942. https://doi.org/10.1256/qj.05.235

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_mixnorm(0.0, [0.1, -0.3, 1.0], [0.4, 2.1, 0.7], [0.1, 0.2, 0.7])
    0.46806866729387275
    """
    B = backends.active if backend is None else backends[backend]
    obs, m, s = map(B.asarray, (obs, m, s))

    if w is None:
        M: int = m.shape[m_axis]
        w = B.zeros(m.shape) + 1 / M
    else:
        w = B.asarray(w)
        w = w / B.sum(w, axis=m_axis, keepdims=True)

    if check_pars:
        if B.any(s <= 0):
            raise ValueError(
                "`s` contains non-positive entries. The scale parameters of the normal distributions should be positive."
            )
        if B.any(w < 0):
            raise ValueError("`w` contains negative entries")

    if m_axis != -1:
        m = B.moveaxis(m, m_axis, -1)
        s = B.moveaxis(s, m_axis, -1)
        w = B.moveaxis(w, m_axis, -1)

    return crps.mixnorm(obs, m, s, w, backend=backend)


def crps_negbinom(
    obs: "ArrayLike",
    /,
    n: "ArrayLike",
    prob: "ArrayLike | None" = None,
    *,
    mu: "ArrayLike | None" = None,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the negative binomial distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{n, p}, y) = y (2 F_{n, p}(y) - 1) - \frac{n(1 - p)}{p^{2}} \left( p (2 F_{n+1, p}(y - 1) - 1) + _{2} F_{1} \left( n + 1, \frac{1}{2}; 2; -\frac{4(1 - p)}{p^{2}} \right) \right),

    where :math:`F_{n, p}` is the CDF of the negative binomial distribution with
    size parameter :math:`n > 0` and probability parameter :math:`p \in (0, 1]`. The mean
    of the negative binomial distribution is :math:`\mu = n (1 - p)/p`.

    Parameters
    ----------
    obs : array_like
        The observed values.
    n: array_like
        Size parameter of the forecast negative binomial distribution.
    prob : array_like
        Probability parameter of the forecast negative binomial distribution.
    mu: array_like
        Mean of the forecast negative binomial distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between NegBinomial(n, prob) and obs.

    References
    ----------
    .. [1] Wei, W., Held, L. (2014),
        Calibration tests for count data.
        TEST 23, 787-805. https://doi.org/10.1007/s11749-014-0380-8

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_negbinom(2, 5, 0.5)
    1.5533629909058577

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

    return crps.negbinom(obs, n, prob, backend=backend)


def crps_normal(
    obs: "ArrayLike",
    /,
    mu: "ArrayLike" = 0.0,
    sigma: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the normal distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(\mathcal{N}(\mu, \sigma), y) = \sigma \Bigl\{ \omega [\Phi(ω) - 1] + 2 \phi(\omega) - \frac{1}{\sqrt{\pi}} \Bigl\}

    where :math:`\Phi(ω)` and :math:`\phi(ω)` are respectively the CDF and PDF of the standard normal
    distribution at the normalized prediction error :math:`\omega = \frac{y - \mu}{\sigma}`.

    Parameters
    ----------
    obs : array_like
        The observed values.
    mu: array_like
        Mean of the forecast normal distribution.
    sigma: array_like
        Standard deviation of the forecast normal distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between Normal(mu, sigma) and obs.

    References
    ----------
    .. [1] Gneiting, T., A. E. Raftery, A. H. Westveld, and T. Goldman (2005),
        Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation.
        Mon. Wea. Rev., 133, 1098-1118, https://doi.org/10.1175/MWR2904.1.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_normal(0.0, 0.1, 0.4)
    0.10339992515976162
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        sigma = B.asarray(sigma)
        if B.any(sigma <= 0):
            raise ValueError(
                "`sigma` contains non-positive entries. The standard deviation of the normal distribution must be positive."
            )
    return crps.normal(obs, mu, sigma, backend=backend)


def crps_2pnormal(
    obs: "ArrayLike",
    /,
    scale1: "ArrayLike" = 1.0,
    scale2: "ArrayLike" = 1.0,
    location: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the two-piece normal distribution.

    It is based on the following relationship given in [1]_:

    .. math::
        \mathrm{CRPS}(F_{\sigma_{1}, \sigma_{2}, \mu}, y) =
        \sigma_{1} \mathrm{CRPS} \left( F_{-\infty,0}^{0, \sigma_{2}/(\sigma_{1} + \sigma_{2})}, \frac{\min(0, y - \mu)}{\sigma_{1}} \right) +
        \sigma_{2} \mathrm{CRPS} \left( F_{0, \sigma_{1}/(\sigma_{1} + \sigma_{2})}^{\infty, 0}, \frac{\min(0, y - \mu)}{\sigma_{2}} \right),

    where :math:`F_{\sigma_{1}, \sigma_{2}, \mu}` is the two-piece normal distribution with
    scale1 and scale2 parameters :math:`\sigma_{1}, \sigma_{2} > 0` and location parameter :math:`\mu`,
    and :math:`F_{l, L}^{u, U}` is the CDF of the generalised truncated and censored normal distribution.

    Parameters
    ----------
    obs : array_like
        The observed values.
    scale1 : array_like
        Scale parameter of the lower half of the forecast two-piece normal distribution.
    scale2 : array_like
        Scale parameter of the upper half of the forecast two-piece normal distribution.
    mu: array_like
        Location parameter of the forecast two-piece normal distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between 2pNormal(scale1, scale2, mu) and obs.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_2pnormal(0.0, 0.4, 2.0, 0.1)
    0.7243199144002115
    """
    B = backends.active if backend is None else backends[backend]
    obs, scale1, scale2, location = map(B.asarray, (obs, scale1, scale2, location))
    if check_pars:
        if B.any(scale1 <= 0):
            raise ValueError(
                "`scale1` contains non-positive entries. The scale parameters of the two-piece normal distribution must be positive."
            )
        if B.any(scale2 <= 0):
            raise ValueError(
                "`scale2` contains non-positive entries. The scale parameters of the two-piece normal distribution must be positive."
            )
    lower = float("-inf")
    upper = 0.0
    lmass = 0.0
    umass = scale2 / (scale1 + scale2)
    z = B.minimum(B.asarray(0.0), B.asarray(obs - location)) / scale1
    s1 = scale1 * crps.gtcnormal(
        z, 0.0, 1.0, lower, upper, lmass, umass, backend=backend
    )
    lower = 0.0
    upper = float("inf")
    lmass = scale1 / (scale1 + scale2)
    umass = 0.0
    z = B.maximum(B.asarray(0.0), B.asarray(obs - location)) / scale2
    s2 = scale2 * crps.gtcnormal(
        z, 0.0, 1.0, lower, upper, lmass, umass, backend=backend
    )
    return s1 + s2


def crps_poisson(
    obs: "ArrayLike",
    /,
    mean: "ArrayLike",
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the Poisson distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F_{\lambda}, y) = (y - \lambda) (2F_{\lambda}(y) - 1)
        + 2 \lambda f_{\lambda}(\lfloor y \rfloor )
        - \lambda \exp (-2 \lambda) (I_{0} (2 \lambda) + I_{1} (2 \lambda)),

    where :math:`F_{\lambda}` is Poisson distribution function with mean parameter :math:`\lambda > 0`,
    and :math:`I_{0}` and :math:`I_{1}` are modified Bessel functions of the first kind.

    Parameters
    ----------
    obs : array_like
        The observed values.
    mean : array_like
        Mean parameter of the forecast poisson distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between Pois(mean) and obs.

    References
    ----------
    .. [1] Wei, W., Held, L. (2014),
        Calibration tests for count data.
        TEST 23, 787-805. https://doi.org/10.1007/s11749-014-0380-8

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_poisson(1, 2)
    0.4991650450203817
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        mean = B.asarray(mean)
        if B.any(mean <= 0):
            raise ValueError(
                "`mean` contains non-positive entries. The mean parameter of the Poisson distribution must be positive."
            )
    return crps.poisson(obs, mean, backend=backend)


def crps_t(
    obs: "ArrayLike",
    /,
    df: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the student's t distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(F, y) = \sigma \left\{ \omega (2 F_{\nu} (\omega) - 1)
        + 2 f_{\nu} \left( \frac{\nu + \omega^{2}}{\nu - 1} \right)
        - \frac{2 \sqrt{\nu}}{\nu - 1} \frac{B(\frac{1}{2},
        \nu - \frac{1}{2})}{B(\frac{1}{2}, \frac{\nu}{2}^{2})}  \right\},

    where :math:`\omega = (y - \mu)/\sigma`, where :math:`\nu > 1, \mu`, and :math:`\sigma > 0` are the
    degrees of freedom, location, and scale parameters respectively of the Student's t
    distribution, and :math:`f_{\nu}` and :math:`F_{\nu}` are the PDF and CDF of the standard Student's
    t distribution with :math:`\nu` degrees of freedom.

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
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between t(df, location, scale) and obs.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_t(0.0, 0.1, 0.4, 0.1)
    0.07687151141732129
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
    return crps.t(obs, df, location, scale, backend=backend)


def crps_uniform(
    obs: "ArrayLike",
    /,
    min: "ArrayLike" = 0.0,
    max: "ArrayLike" = 1.0,
    lmass: "ArrayLike" = 0.0,
    umass: "ArrayLike" = 0.0,
    *,
    backend: "Backend" = None,
    check_pars: bool = False,
) -> "ArrayLike":
    r"""Compute the closed form of the CRPS for the uniform distribution.

    It is based on the following formulation from [1]_:

    .. math::
        \mathrm{CRPS}(\mathcal{U}_{L}^{U}(l, u), y) = (u - l) \left\{ | \frac{y - l}{u - l}
        - F \left( \frac{y - l}{u - l} \right) |
        + F \left( \frac{y - l}{u - l} \right)^{2} (1 - L - U)
        - F \left( \frac{y - l}{u - l} \right) (1 - 2L) + \frac{(1 - L - U)^{2}}{3} + (1 - L)U \right\}

    where :math:`\mathcal{U}_{L}^{U}(l, u)` is the uniform distribution with lower bound :math:`l`,
    upper bound :math:`u > l`, point mass :math:`L` on the lower bound, and point mass :math:`U` on the upper bound.
    We must have that :math:`L, U \ge 0, L + U < 1`.

    Parameters
    ----------
    obs : array_like
        The observed values.
    min: array_like
        Lower bound of the forecast uniform distribution.
    max: array_like
        Upper bound of the forecast uniform distribution.
    lmass : array_like
        Point mass on the lower bound of the forecast uniform distribution.
    umass : array_like
        Point mass on the upper bound of the forecast uniform distribution.
    check_pars: bool
        Boolean indicating whether distribution parameter checks should be carried out prior to implementation.
        Default is False.

    Returns
    -------
    crps : array_like
        The CRPS between U(min, max, lmass, umass) and obs.

    References
    ----------
    .. [1] Jordan, A., Krüger, F., & Lerch, S. (2019).
        Evaluating Probabilistic Forecasts with scoringRules.
        Journal of Statistical Software, 90(12), 1-37.
        https://doi.org/10.18637/jss.v090.i12

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.crps_uniform(0.4, 0.0, 1.0, 0.0, 0.0)
    0.09333333333333332
    """
    if check_pars:
        B = backends.active if backend is None else backends[backend]
        lmass, umass, min, max = map(B.asarray, (lmass, umass, min, max))
        if B.any(lmass < 0) or B.any(lmass > 1):
            raise ValueError("`lmass` contains entries outside the range [0, 1].")
        if B.any(umass < 0) or B.any(umass > 1):
            raise ValueError("`umass` contains entries outside the range [0, 1].")
        if B.any(umass + lmass >= 1):
            raise ValueError("The sum of `umass` and `lmass` should be smaller than 1.")
        if B.any(min >= max):
            raise ValueError("`min` is not always smaller than `max`.")
    return crps.uniform(obs, min, max, lmass, umass, backend=backend)


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
    "crps_2pexponential",
    "crps_gamma",
    "crps_gev",
    "crps_gpd",
    "crps_gtclogistic",
    "crps_tlogistic",
    "crps_clogistic",
    "crps_gtcnormal",
    "crps_tnormal",
    "crps_cnormal",
    "crps_gtct",
    "crps_tt",
    "crps_ct",
    "crps_hypergeometric",
    "crps_laplace",
    "crps_logistic",
    "crps_loglaplace",
    "crps_loglogistic",
    "crps_lognormal",
    "crps_mixnorm",
    "crps_negbinom",
    "crps_normal",
    "crps_2pnormal",
    "crps_poisson",
    "crps_t",
    "crps_uniform",
]
