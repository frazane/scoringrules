import typing as tp

import numpy as np

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
):
    """Plot the reliability diagram of a set of predictions.

    CORP: Consistent, Optimally binned, Reproducible, PAV-algorithm based
    reliability diagram from
    [Dimitriadis et al. (2021)](https://www.pnas.org/doi/full/10.1073/pnas.2016191118)

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    uncertainty_band: str or None
        The type of uncertainty band to plot. If None, no uncertainty band is plotted.
    n_bootstrap: int
        The number of bootstrap samples to use for the uncertainty band.
    alpha: float
        The confidence level for the uncertainty band.

    Returns
    -------
    - plt.Figure
        The reliability diagram plot.
    - plt.Axes
        The axes of the reliability diagram plot.
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

    # CORP reliability via isotonic regression
    prob_argsort = np.argsort(forecasts)
    x = forecasts[prob_argsort]
    y = observations[prob_argsort]
    cep = isotonic_regression(y).x

    # (mis)calibration metrics
    sc = np.mean((y - cep) ** 2)
    sx = np.mean((y - x) ** 2)
    sr = np.mean((y - np.mean(y)) ** 2)

    # uncertainty quantification
    if uncertainty_band is not None:
        ql, qu = _uncertainty_band(x, cep, n_bootstrap, uncertainty_band, alpha)

    # figure
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, cep, "r|")
    ax.plot(x, cep, "r", lw=0.2)
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


def _uncertainty_band(x, cep, n_bootstrap=100, bandtype="consistency", alpha=0.05):
    N = x.size
    M = n_bootstrap
    res = []
    for _ in range(M):
        _idx_resample = np.random.choice(np.arange(N), N, replace=True)
        _x = x[_idx_resample]
        if bandtype == "confidence":
            _y = bernoulli.rvs(_x)
        elif bandtype == "consistency":
            _y = bernoulli.rvs(cep[_idx_resample])
        _x_argsort = np.argsort(_x)
        _x = _x[_x_argsort]
        _y = _y[_x_argsort]
        _cep = isotonic_regression(_y).x
        res.append(
            interp1d(
                _x, _cep, fill_value="nan", bounds_error=False, assume_sorted=True
            )(x)
        )
    res = np.array(res)
    ql = np.nanpercentile(res, alpha * 100, axis=0)
    qu = np.nanpercentile(res, (1 - alpha) * 100, axis=0)
    return ql, qu
