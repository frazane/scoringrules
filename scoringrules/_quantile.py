import typing as tp

from scoringrules._dispatch import resolve_backend_arg
from scoringrules.backend import get_namespace
from scoringrules.core import quantile

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def quantile_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    alpha: "ArrayLike",
    *,
    backend: "Backend | None" = None,
) -> "Array":
    r"""Compute the quantile score for a given quantile level.

    The quantile score (Koenker, R. and G. Bassett, 1978) is defined as

    .. math::
        S_{\alpha}(q_{\alpha}, y) = \begin{cases}
        (1 - \alpha) (q_{\alpha} - y), & \text{if } y \leq q_{\alpha}, \\
        \alpha (y - q_{\alpha}), & \text{if } y > q_{\alpha}.
        \end{cases}

    where :math:`y` is the observed value and :math:`q_{\alpha}` is the predicted value at the
    :math:`\alpha` quantile level.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The forecast values.
    alpha : array_like
        The quantile level.
    backend : str, optional
        Computational backend. ``None`` (default) infers the array framework from
        the inputs. This score has no numba fast path, so ``"numba"`` also infers.

    Returns
    -------
    score : array_like
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
    resolve_backend_arg(backend)
    xp = get_namespace(obs, fct, alpha)
    return quantile.quantile_score(obs, fct, alpha, xp=xp)
