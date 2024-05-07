import typing as tp

from scoringrules.backend import backends
from scoringrules.core import variogram
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def variogram_score(
    observations: "Array",
    forecasts: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    p: float = 1.0,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Variogram Score for a finite multivariate ensemble.

    For a $D$-variate ensemble the Variogram Score
    [(Sheuerer and Hamill, 2015)](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml#bib9)
    of order $p$ is expressed as

    $$\text{VS}_{p}(F_{ens}, \mathbf{y})= \sum_{i=1}^{d} \sum_{j=1}^{d}
    \left( \frac{1}{M} \sum_{m=1}^{M} | x_{m,i} - x_{m,j} |^{p} - | y_{i} - y_{j} |^{p} \right)^{2}. $$

    where $\mathbf{X}$ and $\mathbf{X'}$ are independently sampled ensembles from from $F$.


    Parameters
    ----------
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    p: float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    variogram_score: Array
        The computed Variogram Score.
    """
    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    if backend == "numba":
        return variogram._variogram_score_gufunc(observations, forecasts, p)

    return variogram.vs(observations, forecasts, p, backend=backend)


def twvariogram_score(
    observations: "Array",
    forecasts: "Array",
    v_func: tp.Callable,
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    p: float = 1.0,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Threshold-Weighted Variogram Score (twVS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the twVS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    \[
        \mathrm{twVS}(F_{ens}, \mathbf{y}) = \sum_{i,j=1}^{D}(|v(\mathbf{y})_i - v(\mathbf{y})_{j}|^{p} - \frac{1}{M} \sum_{m=1}^{M}|v(\mathbf{x}_{m})_{i} - v(\mathbf{x}_{m})_{j}|^{p})^{2},
    \]

    where $F_{ens}$ is the ensemble forecast $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$ with
    $M$ members, and $v$ is the chaining function used to target particular outcomes.

    Parameters
    ----------
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    p: float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    v_func: tp.Callable
        Chaining function used to emphasise particular outcomes.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.


    Returns
    -------
    twvariogram_score: ArrayLike of shape (...)
        The computed Threshold-Weighted Variogram Score.
    """
    observations, forecasts = map(v_func, (observations, forecasts))
    return variogram_score(
        observations, forecasts, m_axis, v_axis, p=p, backend=backend
    )


def owvariogram_score(
    observations: "Array",
    forecasts: "Array",
    w_func: tp.Callable,
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    p: float = 1.0,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Outcome-Weighted Variogram Score (owVS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the owVS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    \[
    \begin{split}
        \mathrm{owVS}(F_{ens}, \mathbf{y}) = & \frac{1}{M \bar{w}} \sum_{m=1}^{M} \sum_{i,j=1}^{D}(|y_{i} - y_{j}|^{p} - |x_{m,i} - x_{m,j}|^{p})^{2} w(\mathbf{x}_{m}) w(\mathbf{y}) \\
            & - \frac{1}{2 M^{2} \bar{w}^{2}} \sum_{k,m=1}^{M} \sum_{i,j=1}^{D} \left( |x_{k,i} - x_{k,j}|^{p} - |x_{m,i} - x_{m,j}|^{p} \right)^{2} w(\mathbf{x}_{k}) w(\mathbf{x}_{m}) w(\mathbf{y}),
    \end{split}
    \]

    where $F_{ens}$ is the ensemble forecast $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$ with
    $M$ members, $w$ is the chosen weight function, and $\bar{w} = \sum_{m=1}^{M}w(\mathbf{x}_{m})/M$.

    Parameters
    ----------
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    p: float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    owvariogram_score: ArrayLike of shape (...)
        The computed Outcome-Weighted Variogram Score.
    """
    B = backends.active if backend is None else backends[backend]

    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    obs_weights = B.apply_along_axis(w_func, observations, -1)
    fct_weights = B.apply_along_axis(w_func, forecasts, -1)

    if backend == "numba":
        return variogram._owvariogram_score_gufunc(
            observations, forecasts, p, obs_weights, fct_weights
        )

    return variogram.owvs(
        observations, forecasts, obs_weights, fct_weights, p=p, backend=backend
    )


def vrvariogram_score(
    observations: "Array",
    forecasts: "Array",
    w_func: tp.Callable,
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    p: float = 1.0,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Vertically Re-scaled Variogram Score (vrVS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the vrVS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    \[
    \begin{split}
        \mathrm{vrVS}(F_{ens}, \mathbf{y}) = & \frac{1}{M} \sum_{m=1}^{M} \sum_{i,j=1}^{D}(|y_{i} - y_{j}|^{p} - |x_{m,i} - x_{m,j}|^{p})^{2} w(\mathbf{x}_{m}) w(\mathbf{y}) \\
            & - \frac{1}{2 M^{2}} \sum_{k,m=1}^{M} \sum_{i,j=1}^{D} \left( |x_{k,i} - x_{k,j}|^{p} - |x_{m,i} - x_{m,j}|^{p} \right)^{2} w(\mathbf{x}_{k}) w(\mathbf{x}_{m})) \\
            & + \left( \frac{1}{M} \sum_{m = 1}^{M} \sum_{i,j=1}^{D}(|x_{m,i} - x_{m,j}|^{p} w(\mathbf{x}_{m}) - \sum_{i,j=1}^{D}(|y_{i} - y_{j}|^{p} w(\mathbf{y}) \right) \left( \frac{1}{M} \sum_{m = 1}^{M} w(\mathbf{x}_{m}) - w(\mathbf{y}) \right),
    \end{split}
    \]

    where $F_{ens}$ is the ensemble forecast $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$ with
    $M$ members, $w$ is the chosen weight function, and $\bar{w} = \sum_{m=1}^{M}w(\mathbf{x}_{m})/M$.

    Parameters
    ----------
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    p: float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    vrvariogram_score: ArrayLike of shape (...)
        The computed Vertically Re-scaled Variogram Score.
    """
    B = backends.active if backend is None else backends[backend]

    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    obs_weights = B.apply_along_axis(w_func, observations, -1)
    fct_weights = B.apply_along_axis(w_func, forecasts, -1)

    if backend == "numba":
        return variogram._vrvariogram_score_gufunc(
            observations, forecasts, p, obs_weights, fct_weights
        )

    return variogram.vrvs(
        observations, forecasts, obs_weights, fct_weights, p=p, backend=backend
    )
