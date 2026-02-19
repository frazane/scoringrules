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
    elif estimator == "qd":
        out = _crps_ensemble_qd(obs, fct, backend=backend)
    elif estimator == "akr":
        out = _crps_ensemble_akr(obs, fct, backend=backend)
    elif estimator == "akr_circperm":
        out = _crps_ensemble_akr_circperm(obs, fct, backend=backend)
    elif estimator == "int":
        out = _crps_ensemble_int(obs, fct, backend=backend)
    else:
        raise ValueError(
            f"{estimator} not a valid estimator, must be one of 'nrg', 'fair', 'pwm', 'qd', 'akr', 'akr_circperm' and 'int'."
        )
    return out


def _crps_ensemble_fair(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """Fair version of the CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M = fct.shape[-1]
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


def _crps_ensemble_akr(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the approximate kernel representation."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = B.sum(B.abs(fct - B.roll(fct, shift=1, axis=-1)), -1) / M
    return e_1 - 0.5 * e_2


def _crps_ensemble_akr_circperm(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the AKR with cyclic permutation."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
    shift = M // 2
    e_2 = B.sum(B.abs(fct - B.roll(fct, shift=shift, axis=-1)), -1) / M
    return e_1 - 0.5 * e_2


def _crps_ensemble_qd(obs: "Array", fct: "Array", backend: "Backend" = None) -> "Array":
    """CRPS estimator based on the quantile decomposition form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    alpha = B.arange(1, M + 1) - 0.5
    below = (fct <= obs[..., None]) * alpha * (obs[..., None] - fct)
    above = (fct > obs[..., None]) * (M - alpha) * (fct - obs[..., None])
    out = B.sum(below + above, axis=-1) / (M**2)
    return 2 * out


def _crps_ensemble_akr(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the approximate kernel representation."""
    B = backends.active if backend is None else backends[backend]
    M = fct.shape[-1]
    e_1 = B.mean(B.abs(obs[..., None] - fct), axis=-1)
    ind = [(i + 1) % M for i in range(M)]
    e_2 = B.mean(B.abs(fct[..., ind] - fct)[..., ind], axis=-1)
    return e_1 - 0.5 * e_2


def _crps_ensemble_akr_circperm(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the AKR with cyclic permutation."""
    B = backends.active if backend is None else backends[backend]
    M = fct.shape[-1]
    e_1 = B.mean(B.abs(obs[..., None] - fct), axis=-1)
    shift = int((M - 1) / 2)
    ind = [(i + shift) % M for i in range(M)]
    e_2 = B.mean(B.abs(fct[..., ind] - fct), axis=-1)
    return e_1 - 0.5 * e_2


def _crps_ensemble_int(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the integral representation."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    y_pos = B.mean((fct <= obs[..., None]) * 1.0, axis=-1, keepdims=True)
    fct_cdf = B.zeros(fct.shape) + B.arange(1, M + 1) / M
    fct_cdf = B.concat((fct_cdf, y_pos), axis=-1)
    fct_cdf = B.sort(fct_cdf, axis=-1)
    fct_exp = B.concat((fct, obs[..., None]), axis=-1)
    fct_exp = B.sort(fct_exp, axis=-1)
    fct_dif = fct_exp[..., 1:] - fct_exp[..., :M]
    obs_cdf = (obs[..., None] <= fct_exp) * 1.0
    out = fct_dif * (fct_cdf[..., :M] - obs_cdf[..., :M]) ** 2
    return B.sum(out, axis=-1)


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
    """Outcome-Weighted CRPS for an ensemble forecast."""
    B = backends.active if backend is None else backends[backend]
    wbar = B.mean(fw, axis=-1)
    e_1 = B.mean(B.abs(obs[..., None] - fct) * fw, axis=-1) * ow / wbar
    e_2 = B.mean(
        B.abs(fct[..., None] - fct[..., None, :]) * fw[..., None] * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 *= ow / (wbar**2)
    return e_1 - 0.5 * e_2


def vr_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Vertically Re-scaled CRPS for an ensemble forecast."""
    B = backends.active if backend is None else backends[backend]
    e_1 = B.mean(B.abs(obs[..., None] - fct) * fw, axis=-1) * ow
    e_2 = B.mean(
        B.abs(fct[..., None] - fct[..., None, :]) * fw[..., None] * fw[..., None, :],
        axis=(-1, -2),
    )
    e_3 = B.mean(B.abs(fct) * fw, axis=-1) - B.abs(obs) * ow
    e_3 *= B.mean(fw, axis=1) - ow
    return e_1 - 0.5 * e_2 + e_3
