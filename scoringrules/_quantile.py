import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def quantile_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    alpha: "ArrayLike",
    backend: "Backend | None" = None,
) -> "Array":
    r"""Compute the quantile score for a given quantile level.

    The quantile score (Koenker, R. and G. Bassett, 1978) is defined as

    $$
        S_{\alpha}(q_{\alpha}, y) = \begin{cases}
        (1 - \alpha) (q_{\alpha} - y), & \text{if } y \leq q_{\alpha}, \\
        \alpha (y - q_{\alpha}), & \text{if } y > q_{\alpha}.
        \end{cases}
    $$

    where $y$ is the observed value and $q_{\alpha}$ is the predicted value at the
    $\alpha$ quantile level.

    Parameters
    ----------
    obs:
        The observed values.
    fct:
        The forecast values.
    alpha:
        The quantile level.

    Returns
    -------
    score:
        The quantile score.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.quantile_score(0.3, 0.5, 0.2)
    0.16

    Raises
    ------
    ValueError
        If the quantile level is not between 0 and 1.
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct, alpha = map(B.asarray, (obs, fct, alpha))
    if B.any(alpha < 0) or B.any(alpha > 1):
        raise ValueError("The quantile level must be between 0 and 1.")
    return (B.asarray((obs < fct), dtype=alpha.dtype) - alpha) * (fct - obs)
