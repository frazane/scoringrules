import typing as tp

from scoringrules.backend import backends
from scoringrules.core import energy
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def energy_score(
    forecasts: "Array",
    observations: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
) -> "Array":
    r"""Compute the Energy Score for a finite multivariate ensemble.

    The Energy Score is a multivariate scoring rule expressed as

    $$ES(F, \mathbf{y}) = E_{F} ||\mathbf{X} - \mathbf{y}||
    - \frac{1}{2}E_{F} ||\mathbf{X} - \mathbf{X'}||,$$

    where $\mathbf{X}$ and $\mathbf{X'}$ are independent samples from $F$
    and $||\cdot||$ is the euclidean norm over the input dimensions (the variables).


    Parameters
    ----------
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    m_axis: int
        The axis corresponding to the ensemble dimension on the forecasts array. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension on the forecasts array (or the observations
        array with an extra dimension on `m_axis`). Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    energy_score: Array of shape (...)
        The computed Energy Score.
    """
    backend = backend if backend is not None else backends._active
    forecasts, observations = multivariate_array_check(
        forecasts, observations, m_axis, v_axis, backend=backend
    )

    if backend == "numba":
        return energy._energy_score_gufunc(forecasts, observations)

    return energy.nrg(forecasts, observations, backend=backend)


def twenergy_score(
    forecasts: "Array",
    observations: "Array",
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
) -> "Array":
    r"""Compute the Threshold-Weighted Energy Score (twES) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the twES in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    \[
        \mathrm{twES}(F_{ens}, \mathbf{y}) = \frac{1}{M} \sum_{m = 1}^{M} \| v(\mathbf{x}_{m}) - v(\mathbf{y}) \| - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} \| v(\mathbf{x}_{m}) - v(\mathbf{x}_{j}) \|,
    \]

    where $F_{ens}$ is the ensemble forecast $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$ with
    $M$ members, $\| \cdotp \|$ is the Euclidean distance, and $v$ is the chaining function
    used to target particular outcomes.


    Parameters
    ----------
    forecasts: ArrayLike of shape (..., M, D)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: ArrayLike of shape (...,D)
        The observed values, where the variables dimension is by default the last axis.
    v_func: tp.Callable
        Chaining function used to emphasise particular outcomes.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int or tuple(int)
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    twenergy_score: ArrayLike of shape (...)
        The computed Threshold-Weighted Energy Score.
    """
    forecasts, observations = map(v_func, (forecasts, observations))
    return energy_score(
        forecasts, observations, m_axis=m_axis, v_axis=v_axis, backend=backend
    )


def owenergy_score(
    forecasts: "Array",
    observations: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
) -> "Array":
    r"""Compute the Outcome-Weighted Energy Score (owES) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the owES in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    \[
        \mathrm{owES}(F_{ens}, \mathbf{y}) = \frac{1}{M \bar{w}} \sum_{m = 1}^{M} \| \mathbf{x}_{m} - \mathbf{y} \| w(\mathbf{x}_{m}) w(\mathbf{y}) - \frac{1}{2 M^{2} \bar{w}^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} \| \mathbf{x}_{m} - \mathbf{x}_{j} \| w(\mathbf{x}_{m}) w(\mathbf{x}_{j}) w(\mathbf{y}),
    \]

    where $F_{ens}$ is the ensemble forecast $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$ with
    $M$ members, $\| \cdotp \|$ is the Euclidean distance, $w$ is the chosen weight function,
    and $\bar{w} = \sum_{m=1}^{M}w(\mathbf{x}_{m})/M$.


    Parameters
    ----------
    forecasts: ArrayLike of shape (..., M, D)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: ArrayLike of shape (...,D)
        The observed values, where the variables dimension is by default the last axis.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
    w_funcargs: tuple
        Additional arguments to the weight function.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int or tuple(int)
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    owenergy_score: ArrayLike of shape (...)
        The computed Outcome-Weighted Energy Score.
    """
    B = backends.active if backend is None else backends[backend]

    forecasts, observations = multivariate_array_check(
        forecasts, observations, m_axis, v_axis, backend=backend
    )

    fcts_weights = B.apply_along_axis(w_func, forecasts, -1)
    obs_weights = B.apply_along_axis(w_func, observations, -1)

    if backend == "numba":
        return energy._owenergy_score_gufunc(
            forecasts, observations, fcts_weights, obs_weights
        )

    return energy.ownrg(
        forecasts, observations, fcts_weights, obs_weights, backend=backend
    )


def vrenergy_score(
    forecasts: "Array",
    observations: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    *,
    m_axis: int = -2,
    v_axis: int = -1,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
) -> "Array":
    r"""Compute the Vertically Re-scaled Energy Score (vrES) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the twES in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    \[
    \begin{split}
        \mathrm{vrES}(F_{ens}, \mathbf{y}) = & \frac{1}{M} \sum_{m = 1}^{M} \| \mathbf{x}_{m} - \mathbf{y} \| w(\mathbf{x}_{m}) w(\mathbf{y}) - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} \| \mathbf{x}_{m} - \mathbf{x}_{j} \| w(\mathbf{x}_{m}) w(\mathbf{x_{j}}) \\
            & + \left( \frac{1}{M} \sum_{m = 1}^{M} \| \mathbf{x}_{m} \| w(\mathbf{x}_{m}) - \| \mathbf{y} \| w(\mathbf{y}) \right) \left( \frac{1}{M} \sum_{m = 1}^{M} w(\mathbf{x}_{m}) - w(\mathbf{y}) \right),
    \end{split}
    \]

    where $F_{ens}$ is the ensemble forecast $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$ with
    $M$ members, and $v$ is the chaining function used to target particular outcomes.


    Parameters
    ----------
    forecasts: ArrayLike of shape (..., M, D)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: ArrayLike of shape (...,D)
        The observed values, where the variables dimension is by default the last axis.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
    w_funcargs: tuple
        Additional arguments to the weight function.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int or tuple(int)
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    vrenergy_score: ArrayLike of shape (...)
        The computed Vertically Re-scaled Energy Score.
    """
    B = backends.active if backend is None else backends[backend]

    forecasts, observations = multivariate_array_check(
        forecasts, observations, m_axis, v_axis, backend=backend
    )

    fcts_weights = B.apply_along_axis(w_func, forecasts, -1)
    obs_weights = B.apply_along_axis(w_func, observations, -1)

    if backend == "numba":
        return energy._vrenergy_score_gufunc(
            forecasts, observations, fcts_weights, obs_weights
        )

    return energy.vrnrg(
        forecasts, observations, fcts_weights, obs_weights, backend=backend
    )
