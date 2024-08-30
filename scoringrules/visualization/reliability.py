import typing as tp

import numpy as np

if tp.TYPE_CHECKING:
    import matplotlib.pyplot as plt

try:
    from scipy.optimize import isotonic_regression

    IR_AVAILABLE = True
except ImportError:
    IR_AVAILABLE = False

from scipy.interpolate import interp1d
from scipy.stats import bernoulli


def reliability_diagram(
    observations: np.ndarray,
    forecasts: np.ndarray,
    /,
    uncertainty_band: tp.Literal["confidence", "consistency"] | None = "consistency",
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    ax: "plt.Axes" = None,
) -> "plt.Axes":
    """Plot the reliability diagram of a set of predictions.

    CORP: Consistent, Optimally binned, Reproducible, PAV-algorithm based
    reliability diagram from
    [Dimitriadis et al. (2021)](https://www.pnas.org/doi/full/10.1073/pnas.2016191118).

    Parameters
    ----------
    observations:
        The observed outcomes, either 0 or 1.
    forecasts:
        Forecasted probabilities between 0 and 1.
    uncertainty_band:
        The type of uncertainty band to plot, which can be either `'confidence'` or
        `'consistency'`band. If None, no uncertainty band is plotted.
    n_bootstrap:
        The number of bootstrap samples to use for the uncertainty band.
    alpha:
        The confidence level for the uncertainty band.

    Returns
    -------
    ax:
        The CORP reliability diagram plot.

    Examples
    --------
    >>> import numpy as np
    >>> from scoringrules.visualization import reliability_diagram
    >>> x = np.random.uniform(0, 1, 1024)
    >>> y = np.random.binomial(1, np.sqrt(x), 1024)
    >>> ax = reliability_diagram(y, x)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting") from e

    if not IR_AVAILABLE:
        raise ImportError(
            "scipy>1.12 is required for isotonic regression, "
            "which is used for the reliability diagram"
        )

    x, y, cep = corp_reliability(observations, forecasts)
    sc, sx, sr = corp_score_decomposition(x, y, cep)

    if uncertainty_band is not None:
        ql, qu = _uncertainty_band(x, cep, n_bootstrap, uncertainty_band, alpha)

    ax = plt.subplot(111)
    ax.plot(x, cep, "r|")
    ax.plot(x, cep, "r", lw=0.2)
    if uncertainty_band is not None:
        ax.fill_between(x, ql, qu, color="lightgrey", alpha=1.0, ec="k", lw=0.5)
    ax.plot([0, 1], [0, 1], "b", lw=0.5)
    ax.set_ylabel("Conditional Event Probability (CEP)")
    ax.set_xlabel("Predicted Probability")
    _textdy = 0.015
    ax.text(0.02, 0.9 + _textdy, f"MCB: {sx - sc:.3f}", fontsize=8, color="red")
    ax.text(0.02, 0.85 + _textdy, f"DSC: {sr - sc:.3f}", fontsize=8)
    ax.text(0.02, 0.8 + _textdy, f"UNC: {sr:.3f}", fontsize=8)
    ax = plt.gca()
    ax.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_aspect("equal")
    ax.grid(True, lw=0.5, ls="--", markevery=0.25, zorder=0)
    ax.set_title("CORP reliability diagram")

    return ax


def corp_reliability(obs, fct):
    """CORP reliability via isotonic regression."""
    prob_argsort = np.argsort(fct)
    x = fct[prob_argsort]
    y = obs[prob_argsort]
    cep = isotonic_regression(y).x
    return x, y, cep


def corp_score_decomposition(x, y, cep):
    """CORP reliability score decomposition."""
    sc = np.mean((y - cep) ** 2)
    sx = np.mean((y - x) ** 2)
    sr = np.mean((y - np.mean(y)) ** 2)
    return sc, sx, sr


def _uncertainty_band(x, cep, n_bootstrap=100, bandtype="consistency", alpha=0.05):
    N = x.size
    M = n_bootstrap
    res = []
    for _ in range(M):
        _idx_resample = np.random.choice(np.arange(N), N, replace=True)
        _x = x[_idx_resample]
        if bandtype == "consistency":
            _y = bernoulli.rvs(_x)
        elif bandtype == "confidence":
            _y = bernoulli.rvs(cep[_idx_resample])
        _x, _y, _cep = corp_reliability(_y, _x)
        res.append(
            interp1d(
                _x, _cep, fill_value="nan", bounds_error=False, assume_sorted=True
            )(x)
        )
    res = np.array(res)
    ql = np.nanpercentile(res, alpha * 100, axis=0)
    qu = np.nanpercentile(res, (1 - alpha) * 100, axis=0)
    return ql, qu
