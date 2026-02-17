import typing as tp

from scoringrules.backend import backends
from scoringrules.core import variogram
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def vs_ensemble(
    obs: "Array",
    fct: "Array",
    /,
    w: "Array" = None,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    p: float = 1.0,
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Variogram Score for a finite multivariate ensemble.

    For a :math:`D`-variate ensemble the Variogram Score [1]_ is defined as:

    .. math::
        \text{VS}_{p}(F_{ens}, \mathbf{y})= \sum_{i=1}^{d} \sum_{j=1}^{d} w_{i,j}
        \left( \frac{1}{M} \sum_{m=1}^{M} | x_{m,i} - x_{m,j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2},

    where :math:`\mathbf{X}` and :math:`\mathbf{X'}` are independently sampled ensembles from from :math:`F`.


    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    w : array_like
        The weights assigned to pairs of dimensions. Must be of shape (..., D, D), where
        D is the dimension, so that the weights are in the last two axes.
    m_axis : int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis : int
        The axis corresponding to the variables dimension. Defaults to -1.
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting.
    p : float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    estimator : str
        The variogram score estimator to be used.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    vs_ensemble : array_like
        The computed Variogram Score.

    References
    ----------
    .. [1] Scheuerer, M., and T. M. Hamill (2015),
        Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities.
        Mon. Wea. Rev., 143, 1321-1334, https://doi.org/10.1175/MWR-D-14-00269.1.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    >>> obs = rng.normal(size=(3, 5))
    >>> fct = rng.normal(size=(3, 10, 5))
    >>> sr.vs_ensemble(obs, fct)
    array([ 8.65630139,  6.84693866, 19.52993307])
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    if w is None:
        D = fct.shape[-1]
        w = B.zeros(obs.shape + (D,)) + 1.0

    if ens_w is None:
        if backend == "numba":
            return variogram._variogram_score_gufunc(obs, fct, w, p)

        return variogram.vs(obs, fct, w, p, backend=backend)

    else:
        ens_w = B.asarray(ens_w)
        if B.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        ens_w = ens_w / B.sum(ens_w, axis=-1, keepdims=True)

        if backend == "numba":
            return variogram._variogram_score_gufunc_w(obs, fct, w, ens_w, p)

        return variogram.vs_w(obs, fct, w, ens_w, p, backend=backend)
    if backend == "numba":
        if estimator == "nrg":
            return variogram._variogram_score_nrg_gufunc(obs, fct, p)
        elif estimator == "fair":
            return variogram._variogram_score_fair_gufunc(obs, fct, p)

    return variogram.vs(obs, fct, p, estimator=estimator, backend=backend)


def twvs_ensemble(
    obs: "Array",
    fct: "Array",
    v_func: tp.Callable,
    /,
    w: "Array" = None,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    p: float = 1.0,
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Threshold-Weighted Variogram Score (twVS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the twVS in [1]_,

    .. math::
        \mathrm{twVS}(F_{ens}, \mathbf{y}) = \sum_{i,j=1}^{D}(|v(\mathbf{y})_i - v(\mathbf{y})_{j}|^{p}
        - \frac{1}{M} \sum_{m=1}^{M}|v(\mathbf{x}_{m})_{i} - v(\mathbf{x}_{m})_{j}|^{p})^{2},

    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, and :math:`v` is the chaining function used to target particular outcomes.

    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    w : array_like
        The weights assigned to pairs of dimensions. Must be of shape (..., D, D), where
        D is the dimension, so that the weights are in the last two axes.
    v_func : callable, array_like -> array_like
        Chaining function used to emphasise particular outcomes.
    m_axis : int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis : int
        The axis corresponding to the variables dimension. Defaults to -1.
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting.
    p : float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    estimator : str
        The variogram score estimator to be used.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    twvs_ensemble : array_like
        The computed Threshold-Weighted Variogram Score.

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
    >>> obs = rng.normal(size=(3, 5))
    >>> fct = rng.normal(size=(3, 10, 5))
    >>> sr.twvs_ensemble(obs, fct, lambda x: np.maximum(x, -0.2))
    array([5.94996894, 4.72029765, 6.08947229])
    """
    obs, fct = map(v_func, (obs, fct))
    return vs_ensemble(
        obs,
        fct,
        w,
        m_axis,
        v_axis,
        ens_w=ens_w,
        p=p,
        estimator=estimator,
        backend=backend,
    )


def owvs_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable,
    /,
    w: "Array" = None,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    p: float = 1.0,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Outcome-Weighted Variogram Score (owVS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the owVS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
        \begin{split}
            \mathrm{owVS}(F_{ens}, \mathbf{y}) = & \frac{1}{M \bar{w}} \sum_{m=1}^{M} \sum_{i,j=1}^{D}(|y_{i} - y_{j}|^{p}
                - |x_{m,i} - x_{m,j}|^{p})^{2} w(\mathbf{x}_{m}) w(\mathbf{y}) \\
                & - \frac{1}{2 M^{2} \bar{w}^{2}} \sum_{k,m=1}^{M} \sum_{i,j=1}^{D} \left( |x_{k,i}
                - x_{k,j}|^{p} - |x_{m,i} - x_{m,j}|^{p} \right)^{2} w(\mathbf{x}_{k}) w(\mathbf{x}_{m}) w(\mathbf{y}),
        \end{split}

    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, :math:`w` is the chosen weight function, and :math:`\bar{w} = \sum_{m=1}^{M}w(\mathbf{x}_{m})/M`.

    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    w : array_like
        The weights assigned to pairs of dimensions. Must be of shape (..., D, D), where
        D is the dimension, so that the weights are in the last two axes.
    m_axis : int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis : int
        The axis corresponding to the variables dimension. Defaults to -1.
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting.
    p : float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    owvs_ensemble : array_like
        The computed Outcome-Weighted Variogram Score.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    >>> obs = rng.normal(size=(3, 5))
    >>> fct = rng.normal(size=(3, 10, 5))
    >>> sr.owvs_ensemble(obs, fct, lambda x: x.mean() + 1.0)
    array([ 9.86816636,  6.75532522, 19.59353723])
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    obs_weights = B.apply_along_axis(w_func, obs, -1)
    fct_weights = B.apply_along_axis(w_func, fct, -1)

    if w is None:
        D = fct.shape[-1]
        w = B.zeros(obs.shape + (D,)) + 1.0

    if ens_w is None:
        if backend == "numba":
            return variogram._owvariogram_score_gufunc(
                obs, fct, w, obs_weights, fct_weights, p
            )

        return variogram.owvs(
            obs, fct, w, obs_weights, fct_weights, p=p, backend=backend
        )

    else:
        ens_w = B.asarray(ens_w)
        if B.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        ens_w = ens_w / B.sum(ens_w, axis=-1, keepdims=True)

        if backend == "numba":
            return variogram._owvariogram_score_gufunc_w(
                obs, fct, w, obs_weights, fct_weights, ens_w, p
            )

        return variogram.owvs_w(
            obs, fct, w, obs_weights, fct_weights, ens_w=ens_w, p=p, backend=backend
        )


def vrvs_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable,
    /,
    w: "Array" = None,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    p: float = 1.0,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Vertically Re-scaled Variogram Score (vrVS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the vrVS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
        \begin{split}
            \mathrm{vrVS}(F_{ens}, \mathbf{y}) = & \frac{1}{M} \sum_{m=1}^{M} \sum_{i,j=1}^{D}(|y_{i} - y_{j}|^{p}
                - |x_{m,i} - x_{m,j}|^{p})^{2} w(\mathbf{x}_{m}) w(\mathbf{y}) \\
                & - \frac{1}{2 M^{2}} \sum_{k,m=1}^{M} \sum_{i,j=1}^{D} \left( |x_{k,i} - x_{k,j}|^{p} - |x_{m,i}
                - x_{m,j}|^{p} \right)^{2} w(\mathbf{x}_{k}) w(\mathbf{x}_{m})) \\
                & + \left( \frac{1}{M} \sum_{m = 1}^{M} \sum_{i,j=1}^{D}(|x_{m,i} - x_{m,j}|^{p} w(\mathbf{x}_{m})
                - \sum_{i,j=1}^{D}(|y_{i} - y_{j}|^{p} w(\mathbf{y}) \right) \left( \frac{1}{M} \sum_{m = 1}^{M} w(\mathbf{x}_{m}) - w(\mathbf{y}) \right),
        \end{split}

    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, :math:`w` is the chosen weight function, and :math:`\bar{w} = \sum_{m=1}^{M}w(\mathbf{x}_{m})/M`.

    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    w : array_like
        The weights assigned to pairs of dimensions. Must be of shape (..., D, D), where
        D is the dimension, so that the weights are in the last two axes.
    m_axis : int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis : int
        The axis corresponding to the variables dimension. Defaults to -1.
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting.
    p : float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    vrvs_ensemble : array_like
        The computed Vertically Re-scaled Variogram Score.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    >>> obs = rng.normal(size=(3, 5))
    >>> fct = rng.normal(size=(3, 10, 5))
    >>> sr.vrvs_ensemble(obs, fct, lambda x: x.max() + 1.0)
    array([46.48256493, 57.90759816, 92.37153472])
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    obs_weights = B.apply_along_axis(w_func, obs, -1)
    fct_weights = B.apply_along_axis(w_func, fct, -1)

    if w is None:
        D = fct.shape[-1]
        w = B.zeros(obs.shape + (D,)) + 1.0

    if ens_w is None:
        if backend == "numba":
            return variogram._vrvariogram_score_gufunc(
                obs, fct, w, obs_weights, fct_weights, p
            )

        return variogram.vrvs(
            obs, fct, w, obs_weights, fct_weights, p=p, backend=backend
        )

    else:
        ens_w = B.asarray(ens_w)
        if B.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        ens_w = ens_w / B.sum(ens_w, axis=-1, keepdims=True)

        if backend == "numba":
            return variogram._vrvariogram_score_gufunc_w(
                obs, fct, w, obs_weights, fct_weights, ens_w, p
            )

        return variogram.vrvs_w(
            obs, fct, w, obs_weights, fct_weights, ens_w=ens_w, p=p, backend=backend
        )
