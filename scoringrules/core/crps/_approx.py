import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def ensemble(
    obs: "ArrayLike",
    fct: "Array",
    ens_w: "Array",
    estimator: str = "pwm",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for a finite ensemble."""
    if estimator == "nrg":
        out = _crps_ensemble_nrg(obs, fct, ens_w, backend=backend)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm(obs, fct, ens_w, backend=backend)
    elif estimator == "fair":
        out = _crps_ensemble_fair(obs, fct, ens_w, backend=backend)
    elif estimator == "qd":
        out = _crps_ensemble_qd(obs, fct, ens_w, backend=backend)
    elif estimator == "akr":
        out = _crps_ensemble_akr(obs, fct, ens_w, backend=backend)
    elif estimator == "akr_circperm":
        out = _crps_ensemble_akr_circperm(obs, fct, ens_w, backend=backend)
    else:
        raise ValueError(
            f"{estimator} can only be used with `numpy` "
            "backend and needs `numba` to be installed"
        )

    return out


def _crps_ensemble_fair(
    obs: "Array", fct: "Array", w: "Array", backend: "Backend" = None
) -> "Array":
    """Fair version of the CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    e_1 = B.sum(B.abs(obs[..., None] - fct) * w, axis=-1)
    e_2 = B.sum(
        B.abs(fct[..., None] - fct[..., None, :]) * w[..., None] * w[..., None, :],
        axis=(-1, -2),
    ) / (1 - B.sum(w * w, axis=-1))
    return e_1 - 0.5 * e_2


def _crps_ensemble_nrg(
    obs: "Array", fct: "Array", w: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    e_1 = B.sum(B.abs(obs[..., None] - fct) * w, axis=-1)
    e_2 = B.sum(
        B.abs(fct[..., None] - fct[..., None, :]) * w[..., None] * w[..., None, :],
        (-1, -2),
    )
    return e_1 - 0.5 * e_2


def _crps_ensemble_pwm(
    obs: "Array", fct: "Array", w: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    B = backends.active if backend is None else backends[backend]
    w_sum = B.cumsum(w, axis=-1)
    expected_diff = B.sum(B.abs(obs[..., None] - fct) * w, axis=-1)
    β_0 = B.sum(fct * w * (1.0 - w), axis=-1)
    β_1 = B.sum(fct * w * (w_sum - w), axis=-1)
    return expected_diff + β_0 - 2.0 * β_1


def _crps_ensemble_qd(
    obs: "Array", fct: "Array", w: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the quantile score decomposition."""
    B = backends.active if backend is None else backends[backend]
    w_sum = B.cumsum(w, axis=-1)
    a = w_sum - 0.5 * w
    dif = fct - obs[..., None]
    c = B.where(dif > 0, 1 - a, -a)
    s = B.sum(w * c * dif, axis=-1)
    return 2 * s


def _crps_ensemble_akr(
    obs: "Array", fct: "Array", w: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the approximate kernel representation."""
    B = backends.active if backend is None else backends[backend]
    M = fct.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fct) * w, axis=-1)
    ind = [(i + 1) % M for i in range(M)]
    e_2 = B.sum(B.abs(fct[..., ind] - fct) * w[..., ind], axis=-1)
    return e_1 - 0.5 * e_2


def _crps_ensemble_akr_circperm(
    obs: "Array", fct: "Array", w: "Array", backend: "Backend" = None
) -> "Array":
    """CRPS estimator based on the AKR with cyclic permutation."""
    B = backends.active if backend is None else backends[backend]
    M = fct.shape[-1]
    e_1 = B.sum(B.abs(obs[..., None] - fct) * w, axis=-1)
    shift = int((M - 1) / 2)
    ind = [(i + shift) % M for i in range(M)]
    e_2 = B.sum(B.abs(fct[..., ind] - fct) * w[..., ind], axis=-1)
    return e_1 - 0.5 * e_2


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
    ens_w: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Outcome-Weighted CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    wbar = B.sum(ens_w * fw, axis=-1)
    e_1 = B.sum(ens_w * B.abs(obs[..., None] - fct) * fw, axis=-1) * ow / wbar
    e_2 = B.sum(
        ens_w[..., None]
        * ens_w[..., None, :]
        * B.abs(fct[..., None] - fct[..., None, :])
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 *= ow / (wbar**2)
    return e_1 - 0.5 * e_2


def vr_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Vertically Re-scaled CRPS estimator based on the energy form."""
    B = backends.active if backend is None else backends[backend]
    e_1 = B.sum(ens_w * B.abs(obs[..., None] - fct) * fw, axis=-1) * ow
    e_2 = B.sum(
        ens_w[..., None]
        * ens_w[..., None, :]
        * B.abs(fct[..., None] - fct[..., None, :])
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_3 = B.sum(ens_w * B.abs(fct) * fw, axis=-1) - B.abs(obs) * ow
    e_3 *= B.sum(ens_w * fw, axis=1) - ow
    return e_1 - 0.5 * e_2 + e_3
