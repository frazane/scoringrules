import typing as tp

from scoringrules.backend import backends
from scoringrules.core import variogram
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def variogram_score(
    forecasts: "Array",
    observations: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    p: float = 1.0,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
) -> "Array":
    r"""Compute the Variogram Score for a finite multivariate ensemble.

    For a $D$-variate ensemble the Variogram Score
    [(Sheuerer and Hamill, 2015)](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml#bib9)
    of order $p$ is expressed as

    $$VS(F, \mathbf{y}) = \sum_{i,j=1}^{D}(|y_i - y_j|^p - E_F|X_i - X_j|^p)^2 ,$$

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
    forecasts, observations = multivariate_array_check(
        forecasts, observations, m_axis, v_axis, backend=backend
    )

    if backend == "numba":
        return variogram._variogram_score_gufunc(forecasts, observations, p)

    return variogram.vs(forecasts, observations, p, backend=backend)


def twvariogram_score(
    forecasts: "Array",
    observations: "Array",
    v_func: tp.Callable,
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    p: float = 1.0,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
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
    v_funcargs: tuple
        Additional arguments to the chaining function.
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
    forecasts, observations = map(v_func, (forecasts, observations))
    return variogram_score(
        forecasts, observations, m_axis, v_axis, p=p, backend=backend
    )


def owvariogram_score(
    forecasts: "Array",
    observations: "Array",
    w_func: tp.Callable,
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    p: float = 1.0,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
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
    w_funcargs: tuple
        Additional arguments to the weight function.
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

    forecasts, observations = multivariate_array_check(
        forecasts, observations, m_axis, v_axis, backend=backend
    )

    fcts_weights = B.apply_along_axis(w_func, forecasts, -1)
    obs_weights = B.apply_along_axis(w_func, observations, -1)

    if backend == "numba":
        return variogram._owvariogram_score_gufunc(
            forecasts, observations, p, fcts_weights, obs_weights
        )

    return variogram.owvs(
        forecasts, observations, fcts_weights, obs_weights, p=p, backend=backend
    )


def vrvariogram_score(
    forecasts: "Array",
    observations: "Array",
    w_func: tp.Callable,
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    p: float = 1.0,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
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
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    p: float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
    w_funcargs: tuple
        Additional arguments to the weight function.
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

    forecasts, observations = multivariate_array_check(
        forecasts, observations, m_axis, v_axis, backend=backend
    )

    fcts_weights = B.apply_along_axis(w_func, forecasts, -1)
    obs_weights = B.apply_along_axis(w_func, observations, -1)

    if backend == "numba":
        return variogram._vrvariogram_score_gufunc(
            forecasts, observations, p, fcts_weights, obs_weights
        )

    return variogram.vrvs(
        forecasts, observations, fcts_weights, obs_weights, p=p, backend=backend
    )
