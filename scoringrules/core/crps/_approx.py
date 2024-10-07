import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def ensemble(
    obs: "ArrayLike",
    fct: "Array",
    estimator: str = "pwm",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for a finite ensemble."""
    if estimator == "nrg":
        out = _crps_ensemble_nrg(obs, fct, backend=backend)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm(obs, fct, backend=backend)
    elif estimator == "fair":
        out = _crps_ensemble_fair(obs, fct, backend=backend)
    else:
        raise ValueError(
            f"{estimator} can only be used with `numpy` "
            "backend and needs `numba` to be installed"
        )

    return out


def _crps_ensemble_fair(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """Fair version of the CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = B.sum(
        B.abs(fct[..., None] - fct[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2


def _crps_ensemble_nrg(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = B.sum(B.abs(fct[..., None] - fct[..., None, :]), (-1, -2)) / (M**2)
    return e_1 - 0.5 * e_2


def _crps_ensemble_pwm(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    expected_diff = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
    β_0 = B.sum(fct, axis=-1) / M
    β_1 = B.sum(fct * B.arange(0, M), axis=-1) / (M * (M - 1.0))
    return expected_diff + β_0 - 2.0 * β_1


def quantile_pinball(
    obs: "Array", fct: "Array", alpha: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS approximation via Pinball Loss."""
    B = backends.active if backend is None else backends[backend]
    below = (fct <= obs[..., None]) * alpha * (obs[..., None] - fct)
    above = (fct > obs[..., None]) * (1 - alpha) * (fct - obs[..., None])
    return 2 * B.mean(below + above, axis=-1)


def ow_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Outcome-Weighted CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    wbar = B.mean(fw, axis=-1)
    e_1 = B.sum(B.abs(obs[..., None] - fct) * fw, axis=-1) * ow / (M * wbar)
    e_2 = B.sum(
        B.abs(fct[..., None] - fct[..., None, :]) * fw[..., None] * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 *= ow / (M**2 * wbar**2)
    return e_1 - 0.5 * e_2


def vr_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Vertically Re-scaled CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fct) * fw, axis=-1) * ow / M
    e_2 = B.sum(
        B.abs(B.expand_dims(fct, axis=-1) - B.expand_dims(fct, axis=-2))
        * (B.expand_dims(fw, axis=-1) * B.expand_dims(fw, axis=-2)),
        axis=(-1, -2),
    ) / (M**2)
    e_3 = B.mean(B.abs(fct) * fw, axis=-1) - B.abs(obs) * ow
    e_3 *= B.mean(fw, axis=1) - ow
    return e_1 - 0.5 * e_2 + e_3
