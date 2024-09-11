import typing as tp

from scoringrules.backend import backends
from scoringrules.core import energy
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def energy_score(
    observations: "Array",
    forecasts: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Energy Score for a finite multivariate ensemble.

    The Energy Score is a multivariate scoring rule expressed as

    $$\text{ES}(F_{ens}, \mathbf{y})= \frac{1}{M} \sum_{m=1}^{M} \| \mathbf{x}_{m} -
      \mathbf{y} \| - \frac{1}{2 M^{2}} \sum_{m=1}^{M} \sum_{j=1}^{M} \| \mathbf{x}_{m} - \mathbf{x}_{j} \| $$

    where $\mathbf{X}$ and $\mathbf{X'}$ are independent samples from $F$
    and $||\cdot||$ is the euclidean norm over the input dimensions (the variables).


    Parameters
    ----------
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
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
    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    if backend == "numba":
        return energy._energy_score_gufunc(observations, forecasts)

    return energy.nrg(observations, forecasts, backend=backend)


def twenergy_score(
    observations: "Array",
    forecasts: "Array",
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: "Backend" = None,
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
    observations: ArrayLike of shape (...,D)
        The observed values, where the variables dimension is by default the last axis.
    forecasts: ArrayLike of shape (..., M, D)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
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
    observations, forecasts = map(v_func, (observations, forecasts))
    return energy_score(
        observations, forecasts, m_axis=m_axis, v_axis=v_axis, backend=backend
    )


def owenergy_score(
    observations: "Array",
    forecasts: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: "Backend" = None,
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
    observations: ArrayLike of shape (...,D)
        The observed values, where the variables dimension is by default the last axis.
    forecasts: ArrayLike of shape (..., M, D)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
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

    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    fct_weights = B.apply_along_axis(w_func, forecasts, -1)
    obs_weights = B.apply_along_axis(w_func, observations, -1)

    if B.name == "numba":
        return energy._owenergy_score_gufunc(
            observations, forecasts, obs_weights, fct_weights
        )

    return energy.ownrg(
        observations, forecasts, obs_weights, fct_weights, backend=backend
    )


def vrenergy_score(
    observations: "Array",
    forecasts: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    *,
    m_axis: int = -2,
    v_axis: int = -1,
    backend: "Backend" = None,
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
    observations: ArrayLike of shape (...,D)
        The observed values, where the variables dimension is by default the last axis.
    forecasts: ArrayLike of shape (..., M, D)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
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

    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    fct_weights = B.apply_along_axis(w_func, forecasts, -1)
    obs_weights = B.apply_along_axis(w_func, observations, -1)

    if backend == "numba":
        return energy._vrenergy_score_gufunc(
            observations, forecasts, obs_weights, fct_weights
        )

    return energy.vrnrg(
        observations, forecasts, obs_weights, fct_weights, backend=backend
    )


def energy_score_iid(
    observations: "Array",
    forecasts: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    shuffle: bool = True,
    seed: int = 42,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Energy Score for a finite multivariate ensemble via the IID estimator.

    The Energy Score is a multivariate scoring rule expressed as

    $$\text{ES}^\text{iid}(F_{ens}, \mathbf{y})= \frac{1}{M} \sum_{m=1}^{M} \| \mathbf{x}_{m} -
      \mathbf{y} \| - \frac{1}{K} \sum_{m=1}^{K} \| \mathbf{x}_{m} - \mathbf{x}_{K+m} \| $$

    where $\mathbf{X}$ are independent samples from $F$, $K = \text{floor}(M / 2)$
    and $||\cdot||$ is the euclidean norm over the input dimensions (the variables).

    To avoid the influence of possible structure during the generation of the samples, we
    shuffle the forecast array once before we calculate the $ES^\text{iid}$.

    Parameters
    ----------
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    m_axis: int
        The axis corresponding to the ensemble dimension on the forecasts array. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension on the forecasts array (or the observations
        array with an extra dimension on `m_axis`). Defaults to -1.
    shuffle: bool
        Whether to shuffle the forecasts once along the ensemble axis `m_axis`.
    seed: int
        The random seed for shuffling.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    energy_score: Array of shape (...)
        The computed Energy Score.
    """
    backend = backend if backend is not None else backends._active

    B = backends.active if backend is None else backends[backend]

    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    if shuffle:
        forecasts = B.shuffle(forecasts, axis=m_axis, seed=seed)

    if backend == "numba":
        return energy._energy_score_iid_gufunc(observations, forecasts)

    return energy.iidnrg(observations, forecasts, backend=backend)


def energy_score_kband(
    observations: "Array",
    forecasts: "Array",
    K: int,
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    shuffle: bool = True,
    seed: int = 42,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Energy Score for a finite multivariate ensemble via the k-Band estimator.

    The Energy Score is a multivariate scoring rule expressed as

    $$\text{ES}^{k-\text{band}}(F_{ens}, \mathbf{y})= \frac{1}{M} \sum_{m=1}^{M} \| \mathbf{x}_{m} -
      \mathbf{y} \| - \frac{1}{MK} \sum_{m=1}^{M} \sum_{k=1}^{K}  \| \mathbf{x}_{m} - \mathbf{x}_{m+k} \| $$

    where $\mathbf{X}$ are independent samples from $F$, $K$ is a user-choosen integer
    and $||\cdot||$ is the euclidean norm over the input dimensions (the variables). Let us note
    that we define $\mathbf{x}_{M+k} = \mathbf{x}_{k}$.

    To avoid the influence of possible structure during the generation of the samples, we
    shuffle the forecast array once before we calculate the $ES^\text{iid}$.

    Parameters
    ----------
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    K : int
        The number of resamples taken to estimate the energy score.
    m_axis: int
        The axis corresponding to the ensemble dimension on the forecasts array. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension on the forecasts array (or the observations
        array with an extra dimension on `m_axis`). Defaults to -1.
    shuffle: bool
        Whether to shuffle the forecasts once along the ensemble axis `m_axis`.
    seed: int
        The random seed for shuffling.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    energy_score: Array of shape (...)
        The computed Energy Score.
    """
    B = backends.active if backend is None else backends[backend]

    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    if shuffle:
        forecasts = B.shuffle(forecasts, axis=m_axis, seed=seed)

    if backend == "numba":
        return energy._energy_score_kband_gufunc(observations, forecasts, K)

    return energy.kbnrg(observations, forecasts, K, backend=backend)
