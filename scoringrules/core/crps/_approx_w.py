import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike  # noqa: F401


def ensemble_w(
    obs: "ArrayLike",
    fct: "Array",
    ens_w: "Array" = None,
    estimator: str = "pwm",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for a finite weighted ensemble."""
    if estimator == "nrg":
        out = _crps_ensemble_nrg_w(obs, fct, ens_w, xp=xp)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm_w(obs, fct, ens_w, xp=xp)
    elif estimator == "fair":
        out = _crps_ensemble_fair_w(obs, fct, ens_w, xp=xp)
    elif estimator == "qd":
        out = _crps_ensemble_qd_w(obs, fct, ens_w, xp=xp)
    elif estimator == "akr":
        out = _crps_ensemble_akr_w(obs, fct, ens_w, xp=xp)
    elif estimator == "akr_circperm":
        out = _crps_ensemble_akr_circperm_w(obs, fct, ens_w, xp=xp)
    elif estimator == "int":
        out = _crps_ensemble_int_w(obs, fct, ens_w, xp=xp)
    else:
        raise ValueError(
            f"{estimator} not a valid estimator, must be one of 'nrg', 'fair', 'pwm', 'qd', 'akr', 'akr_circperm' and 'int'."
        )

    return out


def _crps_ensemble_fair_w(obs: "Array", fct: "Array", w: "Array", *, xp) -> "Array":
    """Fair version of the CRPS estimator based on the energy form."""
    e_1 = xp.sum(xp.abs(obs[..., None] - fct) * w, axis=-1)
    e_2 = xp.sum(
        xp.abs(fct[..., None] - fct[..., None, :]) * w[..., None] * w[..., None, :],
        axis=(-1, -2),
    ) / (1 - xp.sum(w * w, axis=-1))
    return e_1 - 0.5 * e_2


def _crps_ensemble_nrg_w(obs: "Array", fct: "Array", w: "Array", *, xp) -> "Array":
    """CRPS estimator based on the energy form."""
    e_1 = xp.sum(xp.abs(obs[..., None] - fct) * w, axis=-1)
    e_2 = xp.sum(
        xp.abs(fct[..., None] - fct[..., None, :]) * w[..., None] * w[..., None, :],
        axis=(-1, -2),
    )
    return e_1 - 0.5 * e_2


def _crps_ensemble_pwm_w(obs: "Array", fct: "Array", w: "Array", *, xp) -> "Array":
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    w_sum = xp.cumsum(w, axis=-1)
    expected_diff = xp.sum(xp.abs(obs[..., None] - fct) * w, axis=-1)
    β_0 = xp.sum(fct * w, axis=-1)
    β_1 = xp.sum(fct * w * (w_sum - w), axis=-1) / (1 - xp.sum(w * w, axis=-1))
    return expected_diff + β_0 - 2.0 * β_1


def _crps_ensemble_qd_w(obs: "Array", fct: "Array", w: "Array", *, xp) -> "Array":
    """CRPS estimator based on the quantile score decomposition."""
    w_sum = xp.cumsum(w, axis=-1)
    a = w_sum - 0.5 * w
    dif = fct - obs[..., None]
    c = xp.where(dif > 0, 1 - a, -a)
    s = xp.sum(w * c * dif, axis=-1)
    return 2 * s


def _crps_ensemble_akr_w(obs: "Array", fct: "Array", w: "Array", *, xp) -> "Array":
    """CRPS estimator based on the approximate kernel representation."""
    M = fct.shape[-1]
    e_1 = xp.sum(xp.abs(obs[..., None] - fct) * w, axis=-1)
    ind = [(i + 1) % M for i in range(M)]
    e_2 = xp.sum(xp.abs(fct[..., ind] - fct) * w[..., ind], axis=-1)
    return e_1 - 0.5 * e_2


def _crps_ensemble_akr_circperm_w(
    obs: "Array", fct: "Array", w: "Array", *, xp
) -> "Array":
    """CRPS estimator based on the AKR with cyclic permutation."""
    M = fct.shape[-1]
    e_1 = xp.sum(xp.abs(obs[..., None] - fct) * w, axis=-1)
    shift = int((M - 1) / 2)
    ind = [(i + shift) % M for i in range(M)]
    e_2 = xp.sum(xp.abs(fct[..., ind] - fct) * w[..., ind], axis=-1)
    return e_1 - 0.5 * e_2


def _crps_ensemble_int_w(obs: "Array", fct: "Array", w: "Array", *, xp) -> "Array":
    """CRPS estimator based on the integral representation."""
    M: int = fct.shape[-1]
    y_pos = xp.sum((fct <= obs[..., None]) * w, axis=-1, keepdims=True)
    fct_cdf = xp.cumsum(w, axis=-1)
    fct_cdf = xp.concat((fct_cdf, y_pos), axis=-1)
    fct_cdf = xp.sort(fct_cdf, axis=-1)
    fct_exp = xp.concat((fct, obs[..., None]), axis=-1)
    fct_exp = xp.sort(fct_exp, axis=-1)
    fct_dif = fct_exp[..., 1:] - fct_exp[..., :M]
    obs_cdf = (obs[..., None] <= fct_exp) * 1.0
    out = fct_dif * (fct_cdf[..., :M] - obs_cdf[..., :M]) ** 2
    return xp.sum(out, axis=-1)


def ow_ensemble_w(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    *,
    xp,
) -> "Array":
    """Outcome-Weighted CRPS for an ensemble forecast."""
    wbar = xp.sum(ens_w * fw, axis=-1)
    e_1 = xp.sum(ens_w * xp.abs(obs[..., None] - fct) * fw, axis=-1) * ow / wbar
    e_2 = xp.sum(
        ens_w[..., None]
        * ens_w[..., None, :]
        * xp.abs(fct[..., None] - fct[..., None, :])
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 *= ow / (wbar**2)
    return e_1 - 0.5 * e_2


def vr_ensemble_w(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    *,
    xp,
) -> "Array":
    """Vertically Re-scaled CRPS for an ensemble forecast."""
    e_1 = xp.sum(ens_w * xp.abs(obs[..., None] - fct) * fw, axis=-1) * ow
    e_2 = xp.sum(
        ens_w[..., None]
        * ens_w[..., None, :]
        * xp.abs(fct[..., None] - fct[..., None, :])
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_3 = xp.sum(ens_w * xp.abs(fct) * fw, axis=-1) - xp.abs(obs) * ow
    e_3 *= xp.sum(ens_w * fw, axis=-1) - ow
    return e_1 - 0.5 * e_2 + e_3
