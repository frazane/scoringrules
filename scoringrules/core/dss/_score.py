import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def ds_score_uv(
    obs: "Array",
    fct: "Array",
    bias: bool = False,
    backend: "Backend" = None,
) -> "Array":
    """Compute the Dawid Sebastiani Score for a univariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    ens_mean = B.mean(fct, axis=-1)
    sig = B.std(fct, axis=-1, bias=bias)
    bias_precision = ((obs - ens_mean) / sig) ** 2
    log_sig = 2 * B.log(sig)
    return bias_precision + log_sig


def ds_score_mv(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    bias: bool = False,
    backend: "Backend" = None,
) -> "Array":
    """Compute the Dawid Sebastiani Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]

    batch_shape = fct.shape[:-2]
    M, D = fct.shape[-2:]

    fct_flat = B.reshape(fct, (-1, M, D))  # (... M D)
    obs_flat = B.reshape(obs, (-1, D))  # (... D)

    # list to collect scores for each batch
    scores = [
        ds_score_mv_mat(obs_i, fct_i, bias=bias, backend=backend)
        for obs_i, fct_i in zip(obs_flat, fct_flat)
    ]

    # reshape to original batch shape
    return B.reshape(B.stack(scores), batch_shape)


def ds_score_mv_mat(
    obs: "Array",  # (D)
    fct: "Array",  # (M D)
    bias: bool = False,
    backend: "Backend" = None,
) -> "Array":
    """Compute the Dawid Sebastiani Score for one multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    cov = B.cov(fct, rowvar=False, bias=bias)  # (D D)
    precision = B.inv(cov)  # (D D)
    log_det = B.log(B.det(cov))  # ()
    obs_cent = obs - B.mean(fct, axis=-2)  # (D)
    bias_precision = B.squeeze(
        B.expand_dims(obs_cent, axis=-2) @ precision @ B.expand_dims(obs_cent, axis=-1)
    )  # ()
    return bias_precision + log_det
