import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def ensemble(
    fcts: "Array",
    obs: "ArrayLike",
    estimator: str = "pwm",
    backend: str | None = None,
) -> "Array":
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


def _crps_ensemble_fair(
    fcts: "Array", obs: "Array", backend: str | None = None
) -> "Array":
    """Fair version of the CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fcts), axis=-1) / M
    e_2 = B.sum(
        B.abs(fcts[..., None] - fcts[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2


def _crps_ensemble_nrg(
    fcts: "Array", obs: "Array", backend: str | None = None
) -> "Array":
    """CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fcts), axis=-1) / M
    e_2 = B.sum(B.abs(fcts[..., None] - fcts[..., None, :]), (-1, -2)) / (M**2)
    return e_1 - 0.5 * e_2


def _crps_ensemble_pwm(
    fcts: "Array", obs: "Array", backend: str | None = None
) -> "Array":
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    expected_diff = B.sum(B.abs(obs[..., None] - fcts), axis=-1) / M
    β_0 = B.sum(fcts, axis=-1) / M
    β_1 = B.sum(fcts * B.arange(M), axis=-1) / (M * (M - 1))
    return expected_diff + β_0 - 2.0 * β_1


def ow_ensemble(
    fcts: "Array",
    obs: "Array",
    fw: "Array",
    ow: "Array",
    backend: str | None = None,
) -> "Array":
    """Outcome-Weighted CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    wbar = B.mean(fw, axis=-1)
    e_1 = B.sum(B.abs(obs[..., None] - fcts) * fw, axis=-1) * ow / (M * wbar)
    e_2 = B.sum(
        B.abs(fcts[..., None] - fcts[..., None, :]) * fw[..., None] * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 *= ow / (M**2 * wbar**2)
    return e_1 - 0.5 * e_2


def vr_ensemble(
    fcts: "Array",
    obs: "Array",
    fw: "Array",
    ow: "Array",
    backend: str | None = None,
) -> "Array":
    """Vertically Re-scaled CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fcts) * fw, axis=-1) * ow / M
    e_2 = B.sum(
        B.abs(B.expand_dims(fcts, axis=-1) - B.expand_dims(fcts, axis=-2))
        * (B.expand_dims(fw, axis=-1) * B.expand_dims(fw, axis=-2)),
        axis=(-1, -2),
    ) / (M**2)
    e_3 = B.mean(B.abs(fcts) * fw, axis=-1) - B.abs(obs) * ow
    e_3 *= B.mean(fw, axis=1) - ow
    return e_1 - 0.5 * e_2 + e_3
