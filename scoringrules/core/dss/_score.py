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


def ds_score_mv(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    bias: bool = False,
    *,
    xp,
) -> "Array":
    """Compute the Dawid Sebastiani Score for a multivariate finite ensemble.

    Batched over the leading axes: the covariance is the same estimator as
    ``np.cov(fct_i, rowvar=False, bias=bias)`` for every batch element.
    """
    M = fct.shape[-2]
    ens_mean = xp.mean(fct, axis=-2, keepdims=True)  # (... 1 D)
    fct_cent = fct - ens_mean  # (... M D)
    cov = xp.matrix_transpose(fct_cent) @ fct_cent / (M if bias else M - 1)  # (... D D)
    precision = xp.inv(cov)  # (... D D)
    log_det = xp.log(xp.det(cov))  # (...)
    obs_cent = obs - ens_mean[..., 0, :]  # (... D)
    bias_precision = (obs_cent[..., None, :] @ precision @ obs_cent[..., :, None])[
        ..., 0, 0
    ]  # (...)
    return bias_precision + log_det
