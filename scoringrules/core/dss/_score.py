import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def _ds_score(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    backend: "Backend" = None,
) -> "Array":
    """Compute the Dawid Sebastiani score Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    cov = B.cov(fct)
    precision = B.inv(cov)
    log_det = B.log(B.det(cov))
    bias = obs - B.mean(fct, axis=-2)
    bias_precision = B.squeeze(bias[:, None, :] @ precision @ bias[:, :, None])
    return log_det + bias_precision
