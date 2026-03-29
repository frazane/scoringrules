import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def ensemble(
    obs: "ArrayLike",
    fct: "Array",
    estimator: str = "pwm",
    nan_mask=None,
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for a finite ensemble."""
    if estimator == "nrg":
        out = _crps_ensemble_nrg(obs, fct, nan_mask=nan_mask, backend=backend)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm(obs, fct, nan_mask=nan_mask, backend=backend)
    elif estimator == "fair":
        out = _crps_ensemble_fair(obs, fct, nan_mask=nan_mask, backend=backend)
    elif estimator == "qd":
        out = _crps_ensemble_qd(obs, fct, nan_mask=nan_mask, backend=backend)
    elif estimator == "akr":
        out = _crps_ensemble_akr(obs, fct, nan_mask=nan_mask, backend=backend)
    elif estimator == "akr_circperm":
        out = _crps_ensemble_akr_circperm(obs, fct, nan_mask=nan_mask, backend=backend)
    elif estimator == "int":
        out = _crps_ensemble_int(obs, fct, backend=backend)
    else:
        raise ValueError(
            f"{estimator} not a valid estimator, must be one of 'nrg', 'fair', 'pwm', 'qd', 'akr', 'akr_circperm' and 'int'."
        )
    return out


def _crps_ensemble_fair(
    obs: "Array", fct: "Array", nan_mask=None, backend: "Backend" = None
) -> "Array":
    """Fair version of the CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    if nan_mask is not None:
        M_eff = B.sum(B.where(nan_mask, B.asarray(0.0), B.asarray(1.0)), axis=-1)
        e_1 = (
            B.sum(
                B.where(nan_mask, B.asarray(0.0), B.abs(obs[..., None] - fct)), axis=-1
            )
            / M_eff
        )
        pair_mask = nan_mask[..., :, None] | nan_mask[..., None, :]
        e_2 = B.sum(
            B.where(
                pair_mask,
                B.asarray(0.0),
                B.abs(fct[..., None] - fct[..., None, :]),
            ),
            axis=(-1, -2),
        ) / (M_eff * (M_eff - 1))
        result = e_1 - 0.5 * e_2
        return B.where(M_eff <= 1, B.asarray(float("nan")), result)
    else:
        e_1 = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
        e_2 = B.sum(
            B.abs(fct[..., None] - fct[..., None, :]),
            axis=(-1, -2),
        ) / (M * (M - 1))
        return e_1 - 0.5 * e_2


def _crps_ensemble_nrg(
    obs: "Array", fct: "Array", nan_mask=None, backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    if nan_mask is not None:
        M_eff = B.sum(B.where(nan_mask, B.asarray(0.0), B.asarray(1.0)), axis=-1)
        e_1 = (
            B.sum(
                B.where(nan_mask, B.asarray(0.0), B.abs(obs[..., None] - fct)), axis=-1
            )
            / M_eff
        )
        pair_mask = nan_mask[..., :, None] | nan_mask[..., None, :]
        e_2 = (
            B.sum(
                B.where(
                    pair_mask,
                    B.asarray(0.0),
                    B.abs(fct[..., None] - fct[..., None, :]),
                ),
                (-1, -2),
            )
            / M_eff**2
        )
        result = e_1 - 0.5 * e_2
        return B.where(M_eff == 0, B.asarray(float("nan")), result)
    else:
        e_1 = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
        e_2 = B.sum(B.abs(fct[..., None] - fct[..., None, :]), (-1, -2)) / (M**2)
        return e_1 - 0.5 * e_2


def _crps_ensemble_pwm(
    obs: "Array", fct: "Array", nan_mask=None, backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    if nan_mask is not None:
        # Assumes the ensemble is sorted with NaN members zeroed at the end.
        M_eff = B.sum(B.where(nan_mask, B.asarray(0.0), B.asarray(1.0)), axis=-1)
        expected_diff = (
            B.sum(
                B.where(nan_mask, B.asarray(0.0), B.abs(obs[..., None] - fct)), axis=-1
            )
            / M_eff
        )
        # NaN members are zeroed so their contributions to β_0 and β_1 are 0.
        β_0 = B.sum(fct, axis=-1) / M_eff
        β_1 = B.sum(fct * B.arange(0, M), axis=-1) / (M_eff * (M_eff - 1.0))
        result = expected_diff + β_0 - 2.0 * β_1
        return B.where(M_eff <= 1, B.asarray(float("nan")), result)
    else:
        expected_diff = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
        β_0 = B.sum(fct, axis=-1) / M
        β_1 = B.sum(fct * B.arange(0, M), axis=-1) / (M * (M - 1.0))
        return expected_diff + β_0 - 2.0 * β_1


def _crps_ensemble_akr(
    obs: "Array", fct: "Array", nan_mask=None, backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the approximate kernel representation."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    if nan_mask is not None:
        M_eff = B.sum(B.where(nan_mask, B.asarray(0.0), B.asarray(1.0)), axis=-1)
        e_1 = (
            B.sum(
                B.where(nan_mask, B.asarray(0.0), B.abs(obs[..., None] - fct)), axis=-1
            )
            / M_eff
        )
        roll_mask = nan_mask | B.roll(nan_mask, shift=1, axis=-1)
        e_2 = (
            B.sum(
                B.where(
                    roll_mask,
                    B.asarray(0.0),
                    B.abs(fct - B.roll(fct, shift=1, axis=-1)),
                ),
                -1,
            )
            / M_eff
        )
        result = e_1 - 0.5 * e_2
        return B.where(M_eff == 0, B.asarray(float("nan")), result)
    else:
        e_1 = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
        e_2 = B.sum(B.abs(fct - B.roll(fct, shift=1, axis=-1)), -1) / M
        return e_1 - 0.5 * e_2


def _crps_ensemble_akr_circperm(
    obs: "Array", fct: "Array", nan_mask=None, backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the AKR with cyclic permutation."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    if nan_mask is not None:
        M_eff = B.sum(B.where(nan_mask, B.asarray(0.0), B.asarray(1.0)), axis=-1)
        e_1 = (
            B.sum(
                B.where(nan_mask, B.asarray(0.0), B.abs(obs[..., None] - fct)), axis=-1
            )
            / M_eff
        )
        shift = M // 2
        roll_mask = nan_mask | B.roll(nan_mask, shift=shift, axis=-1)
        e_2 = (
            B.sum(
                B.where(
                    roll_mask,
                    B.asarray(0.0),
                    B.abs(fct - B.roll(fct, shift=shift, axis=-1)),
                ),
                -1,
            )
            / M_eff
        )
        result = e_1 - 0.5 * e_2
        return B.where(M_eff == 0, B.asarray(float("nan")), result)
    else:
        e_1 = B.sum(B.abs(obs[..., None] - fct), axis=-1) / M
        shift = M // 2
        e_2 = B.sum(B.abs(fct - B.roll(fct, shift=shift, axis=-1)), -1) / M
        return e_1 - 0.5 * e_2


def _crps_ensemble_qd(
    obs: "Array", fct: "Array", nan_mask=None, backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the quantile decomposition form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    if nan_mask is not None:
        # Assumes the ensemble is sorted with NaN members zeroed at the end.
        M_eff = B.sum(B.where(nan_mask, B.asarray(0.0), B.asarray(1.0)), axis=-1)
        alpha = B.arange(1, M + 1) - 0.5  # shape (M,)
        below = (fct <= obs[..., None]) * alpha * (obs[..., None] - fct)
        above = (
            (fct > obs[..., None]) * (M_eff[..., None] - alpha) * (fct - obs[..., None])
        )
        out = (
            B.sum(B.where(nan_mask, B.asarray(0.0), below + above), axis=-1) / M_eff**2
        )
        result = 2 * out
        return B.where(M_eff == 0, B.asarray(float("nan")), result)
    else:
        alpha = B.arange(1, M + 1) - 0.5
        below = (fct <= obs[..., None]) * alpha * (obs[..., None] - fct)
        above = (fct > obs[..., None]) * (M - alpha) * (fct - obs[..., None])
        out = B.sum(below + above, axis=-1) / (M**2)
        return 2 * out


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
    nan_mask=None,
    backend: "Backend" = None,
) -> "Array":
    """Outcome-Weighted CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    if nan_mask is not None:
        M_eff = B.sum(B.where(nan_mask, B.asarray(0.0), B.asarray(1.0)), axis=-1)
        fw = B.where(nan_mask, B.asarray(0.0), fw)
        wbar = B.sum(fw, axis=-1) / M_eff
        e_1 = (
            B.sum(
                B.where(nan_mask, B.asarray(0.0), B.abs(obs[..., None] - fct) * fw),
                axis=-1,
            )
            * ow
            / (M_eff * wbar)
        )
        pair_mask = nan_mask[..., :, None] | nan_mask[..., None, :]
        e_2 = B.sum(
            B.where(
                pair_mask,
                B.asarray(0.0),
                B.abs(fct[..., None] - fct[..., None, :])
                * fw[..., None]
                * fw[..., None, :],
            ),
            axis=(-1, -2),
        )
        e_2 *= ow / (M_eff**2 * wbar**2)
        result = e_1 - 0.5 * e_2
        return B.where(M_eff == 0, B.asarray(float("nan")), result)
    else:
        wbar = B.mean(fw, axis=-1)
        e_1 = B.sum(B.abs(obs[..., None] - fct) * fw, axis=-1) * ow / (M * wbar)
        e_2 = B.sum(
            B.abs(fct[..., None] - fct[..., None, :])
            * fw[..., None]
            * fw[..., None, :],
            axis=(-1, -2),
        )
        e_2 *= ow / (M**2 * wbar**2)
        return e_1 - 0.5 * e_2


def vr_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    nan_mask=None,
    backend: "Backend" = None,
) -> "Array":
    """Vertically Re-scaled CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    if nan_mask is not None:
        M_eff = B.sum(B.where(nan_mask, B.asarray(0.0), B.asarray(1.0)), axis=-1)
        fw = B.where(nan_mask, B.asarray(0.0), fw)
        e_1 = (
            B.sum(
                B.where(nan_mask, B.asarray(0.0), B.abs(obs[..., None] - fct) * fw),
                axis=-1,
            )
            * ow
            / M_eff
        )
        pair_mask = nan_mask[..., :, None] | nan_mask[..., None, :]
        e_2 = (
            B.sum(
                B.where(
                    pair_mask,
                    B.asarray(0.0),
                    B.abs(B.expand_dims(fct, axis=-1) - B.expand_dims(fct, axis=-2))
                    * (B.expand_dims(fw, axis=-1) * B.expand_dims(fw, axis=-2)),
                ),
                axis=(-1, -2),
            )
            / M_eff**2
        )
        e_3 = (
            B.sum(B.where(nan_mask, B.asarray(0.0), B.abs(fct) * fw), axis=-1) / M_eff
            - B.abs(obs) * ow
        )
        e_3 *= B.sum(fw, axis=-1) / M_eff - ow
        result = e_1 - 0.5 * e_2 + e_3
        return B.where(M_eff == 0, B.asarray(float("nan")), result)
    else:
        e_1 = B.sum(B.abs(obs[..., None] - fct) * fw, axis=-1) * ow / M
        e_2 = B.sum(
            B.abs(B.expand_dims(fct, axis=-1) - B.expand_dims(fct, axis=-2))
            * (B.expand_dims(fw, axis=-1) * B.expand_dims(fw, axis=-2)),
            axis=(-1, -2),
        ) / (M**2)
        e_3 = B.mean(B.abs(fct) * fw, axis=-1) - B.abs(obs) * ow
        e_3 *= B.mean(fw, axis=1) - ow
        return e_1 - 0.5 * e_2 + e_3
