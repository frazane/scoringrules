import typing as tp

from scoringrules.backend import backends as srb
from scoringrules.backend.arrayapi import Array

ArrayLike = TypeVar("ArrayLike", Array, float)


def variogram_score(
    forecasts: Array,
    observations: Array,
    p: float = 1.0,
    m_axis: int = -2,
    v_axis: int = -1,
    backend="numba",
) -> Array:
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
    return srb[backend].variogram_score(
        forecasts, observations, p=p, m_axis=m_axis, v_axis=v_axis
    )




def owvariogram_score(
    forecasts: Array,
    observations: Array,
    p: float = 1.0,
    /,
    w_func: tp.Callable = lambda x, *args: 1.0,
    w_funcargs: tuple = (),
    *,
    m_axis: int = -2,
    v_axis: int = -1,
    backend="numba",
) -> Array:
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
    return srb[backend].owvariogram_score(
        forecasts, observations, p, w_func, w_funcargs, m_axis=m_axis, v_axis=v_axis
    )


def twvariogram_score(
    forecasts: Array,
    observations: Array,
    p: float = 1.0,
    /,
    v_func: tp.Callable = lambda x, *args: x,
    v_funcargs: tuple = (),
    *,
    m_axis: int = -2,
    v_axis: int = -1,
    backend="numba",
) -> Array:
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
    return srb[backend].twvariogram_score(
        forecasts, observations, p, v_func, v_funcargs, m_axis=m_axis, v_axis=v_axis
    )


def vrvariogram_score(
    forecasts: Array,
    observations: Array,
    p: float = 1.0,
    /,
    w_func: tp.Callable = lambda x, *args: 1.0,
    w_funcargs: tuple = (),
    *,
    m_axis: int = -2,
    v_axis: int = -1,
    backend="numba",
) -> Array:
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
    return srb[backend].vrvariogram_score(
        forecasts, observations, p, w_func, w_funcargs, m_axis=m_axis, v_axis=v_axis
    )