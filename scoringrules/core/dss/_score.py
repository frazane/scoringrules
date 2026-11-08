import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def ds_score_uv(
    obs: "Array",
    fct: "Array",
    bias: bool = False,
    *,
    xp,
) -> "Array":
    """Compute the Dawid Sebastiani Score for a univariate finite ensemble."""
    ens_mean = xp.mean(fct, axis=-1)
    sig = xp.std(fct, axis=-1, correction=0 if bias else 1)
    bias_precision = ((obs - ens_mean) / sig) ** 2
    log_sig = 2 * xp.log(sig)
    return bias_precision + log_sig


# A Cholesky pivot below TOL_FACTOR * D * eps times its diagonal entry counts as zero
# (the covariance is singular to the input's precision), and a variable whose
# standard deviation is at most ZERO_VAR_FACTOR * eps times its absolute mean counts
# as constant. eps is the input dtype's. Same constants as in
# ``_gufuncs``.
TOL_FACTOR = 10.0
ZERO_VAR_FACTOR = 1.0


def ds_score_mv(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    bias: bool = False,
    *,
    xp,
) -> "Array":
    """Compute the Dawid Sebastiani Score for a multivariate finite ensemble.

    Batched over the leading axes: the covariance is the same estimator as
    ``np.cov(fct_i, rowvar=False, bias=bias)`` for every batch element. The score
    comes from its Cholesky factor ``L``: ``||L^-1 (y - mean)||^2 + 2 sum log L_jj``.
    It is NaN where the covariance is not positive definite to the input's
    precision (``M <= D``, collinear, constant, NaN or inf members) or ``obs`` is
    not finite, like the gufunc.
    """
    M, D = fct.shape[-2], fct.shape[-1]
    if M <= D:  # the centred members span at most M - 1 dimensions
        return xp.full_like(obs[..., 0], xp.nan)
    ens_mean = xp.mean(fct, axis=-2, keepdims=True)  # (... 1 D)
    fct_cent = fct - ens_mean  # (... M D)
    # refined by the mean residual: a constant variable then has zero variance
    resid = xp.mean(fct_cent, axis=-2, keepdims=True)  # (... 1 D)
    fct_cent = fct_cent - resid
    ens_mean = (ens_mean + resid)[..., 0, :]  # (... D)
    obs_cent = obs - ens_mean  # (... D)
    cov = xp.matrix_transpose(fct_cent) @ fct_cent / (M if bias else M - 1)  # (... D D)
    L = xp.cholesky(cov)  # (... D D), NaN where not positive definite
    eps = xp.finfo(cov.dtype).eps
    var = xp.linalg.diagonal(cov)  # (... D)
    ok = (
        xp.all(xp.linalg.diagonal(L) ** 2 > TOL_FACTOR * D * eps * var, axis=-1)
        & xp.all(xp.sqrt(var) > ZERO_VAR_FACTOR * eps * xp.abs(ens_mean), axis=-1)
        & xp.all(xp.isfinite(obs_cent), axis=-1)
    )  # (...), False for NaN
    eye = xp.eye(D, dtype=cov.dtype, device=xp.device(cov))
    L = xp.where(ok[..., None, None], L, eye)
    z = xp.linalg.solve(L, obs_cent[..., None])[..., 0]  # (... D)
    log_det = 2 * xp.sum(xp.log(xp.linalg.diagonal(L)), axis=-1)  # (...)
    return xp.where(ok, xp.sum(z * z, axis=-1) + log_det, xp.nan)
