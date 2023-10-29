from scoringrules.core.crps.guguncs import estimator_gufuncs
from scoringrules.core.typing import Array, ArrayLike
from scoringrules.backend import _NUMBA_IMPORTED, backends


def ensemble(
    fcts: Array,
    obs: ArrayLike,
    estimator: str = "pwm",
    backend: str | None = None,
) -> Array:
    """Compute the CRPS for a finite ensemble."""


    if estimator == "nrg":
        out = _crps_ensemble_nrg(fcts, obs, backend=backend)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm(fcts, obs, backend=backend)
    elif estimator == "fair":
        out = _crps_ensemble_fair(fcts, obs, backend=backend)
    else:
        raise ValueError(
            f"{estimator} can only be used with `numpy` "
            "backend and needs `numba` to be installed"
        )

    return out

def _crps_ensemble_fair(fcts: Array, obs: Array, backend: str | None = None) -> Array:
    """Fair version of the CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fcts), axis=-1) / M
    e_2 = B.sum(
        B.abs(fcts[..., None] - fcts[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2

def _crps_ensemble_nrg(fcts: Array, obs: Array, backend: str | None = None) -> Array:
    """CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fcts), axis=-1) / M
    e_2 = B.sum(
        B.abs(
            B.expand_dims(fcts, axis=-1) - B.expand_dims(fcts, axis=-2)
        ),
        axis=(-1, -2),
    ) / (M**2)
    return e_1 - 0.5 * e_2

def _crps_ensemble_pwm(fcts: Array, obs: Array, backend: str | None = None) -> Array:
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    expected_diff = B.sum(B.abs(obs[..., None] - fcts), axis=-1) / M
    β_0 = B.sum(fcts, axis=-1) / M
    β_1 = B.sum(fcts * B.arange(M), axis=-1) / (M * (M - 1))
    return expected_diff + β_0 - 2.0 * β_1
