import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike  # noqa: F401


def ensemble(
    obs: "ArrayLike",
    fct: "Array",
    estimator: str = "pwm",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for a finite ensemble."""
    if estimator == "nrg":
        out = _crps_ensemble_nrg(obs, fct, xp=xp)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm(obs, fct, xp=xp)
    elif estimator == "fair":
        out = _crps_ensemble_fair(obs, fct, xp=xp)
    elif estimator == "qd":
        out = _crps_ensemble_qd(obs, fct, xp=xp)
    elif estimator == "akr":
        out = _crps_ensemble_akr(obs, fct, xp=xp)
    elif estimator == "akr_circperm":
        out = _crps_ensemble_akr_circperm(obs, fct, xp=xp)
    elif estimator == "int":
        out = _crps_ensemble_int(obs, fct, xp=xp)
    else:
        raise ValueError(
            f"{estimator} not a valid estimator, must be one of 'nrg', 'fair', 'pwm', 'qd', 'akr', 'akr_circperm' and 'int'."
        )
    return out


def _crps_ensemble_fair(obs: "Array", fct: "Array", *, xp) -> "Array":
    """Fair version of the CRPS estimator based on the energy form."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(xp.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = xp.sum(
        xp.abs(fct[..., None] - fct[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2


def _crps_ensemble_nrg(obs: "Array", fct: "Array", *, xp) -> "Array":
    """CRPS estimator based on the energy form."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(xp.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = xp.sum(xp.abs(fct[..., None] - fct[..., None, :]), axis=(-1, -2)) / (M**2)
    return e_1 - 0.5 * e_2


def _crps_ensemble_pwm(obs: "Array", fct: "Array", *, xp) -> "Array":
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    M: int = fct.shape[-1]
    expected_diff = xp.sum(xp.abs(obs[..., None] - fct), axis=-1) / M
    β_0 = xp.sum(fct, axis=-1) / M
    β_1 = xp.sum(fct * xp.arange(0, M), axis=-1) / (M * (M - 1.0))
    return expected_diff + β_0 - 2.0 * β_1


def _crps_ensemble_akr(obs: "Array", fct: "Array", *, xp) -> "Array":
    """CRPS estimator based on the approximate kernel representation."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(xp.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = xp.sum(xp.abs(fct - xp.roll(fct, shift=1, axis=-1)), axis=-1) / M
    return e_1 - 0.5 * e_2


def _crps_ensemble_akr_circperm(obs: "Array", fct: "Array", *, xp) -> "Array":
    """CRPS estimator based on the AKR with cyclic permutation."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(xp.abs(obs[..., None] - fct), axis=-1) / M
    shift = M // 2
    e_2 = xp.sum(xp.abs(fct - xp.roll(fct, shift=shift, axis=-1)), axis=-1) / M
    return e_1 - 0.5 * e_2


def _crps_ensemble_qd(obs: "Array", fct: "Array", *, xp) -> "Array":
    """CRPS estimator based on the quantile decomposition form."""
    M: int = fct.shape[-1]
    alpha = xp.arange(1, M + 1) - 0.5
    below = (fct <= obs[..., None]) * alpha * (obs[..., None] - fct)
    above = (fct > obs[..., None]) * (M - alpha) * (fct - obs[..., None])
    out = xp.sum(below + above, axis=-1) / (M**2)
    return 2 * out


def _crps_ensemble_int(obs: "Array", fct: "Array", *, xp) -> "Array":
    """CRPS estimator based on the integral representation."""
    M: int = fct.shape[-1]
    y_pos = xp.mean((fct <= obs[..., None]) * 1.0, axis=-1, keepdims=True)
    fct_cdf = xp.zeros(fct.shape) + xp.arange(1, M + 1) / M
    fct_cdf = xp.concat((fct_cdf, y_pos), axis=-1)
    fct_cdf = xp.sort(fct_cdf, axis=-1)
    fct_exp = xp.concat((fct, obs[..., None]), axis=-1)
    fct_exp = xp.sort(fct_exp, axis=-1)
    fct_dif = fct_exp[..., 1:] - fct_exp[..., :M]
    obs_cdf = (obs[..., None] <= fct_exp) * 1.0
    out = fct_dif * (fct_cdf[..., :M] - obs_cdf[..., :M]) ** 2
    return xp.sum(out, axis=-1)


def quantile_pinball(obs: "Array", fct: "Array", alpha: "Array", *, xp) -> "Array":
    """CRPS approximation via Pinball Loss."""
    below = (fct <= obs[..., None]) * alpha * (obs[..., None] - fct)
    above = (fct > obs[..., None]) * (1 - alpha) * (fct - obs[..., None])
    return 2 * xp.mean(below + above, axis=-1)


def ow_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    *,
    xp,
) -> "Array":
    """Outcome-Weighted CRPS estimator based on the energy form."""
    M: int = fct.shape[-1]
    wbar = xp.mean(fw, axis=-1)
    e_1 = xp.sum(xp.abs(obs[..., None] - fct) * fw, axis=-1) * ow / (M * wbar)
    e_2 = xp.sum(
        xp.abs(fct[..., None] - fct[..., None, :]) * fw[..., None] * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 *= ow / (M**2 * wbar**2)
    return e_1 - 0.5 * e_2


def vr_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    *,
    xp,
) -> "Array":
    """Vertically Re-scaled CRPS estimator based on the energy form."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(xp.abs(obs[..., None] - fct) * fw, axis=-1) * ow / M
    e_2 = xp.sum(
        xp.abs(xp.expand_dims(fct, axis=-1) - xp.expand_dims(fct, axis=-2))
        * (xp.expand_dims(fw, axis=-1) * xp.expand_dims(fw, axis=-2)),
        axis=(-1, -2),
    ) / (M**2)
    e_3 = xp.mean(xp.abs(fct) * fw, axis=-1) - xp.abs(obs) * ow
    e_3 *= xp.mean(fw, axis=1) - ow
    return e_1 - 0.5 * e_2 + e_3
