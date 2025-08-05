from ._approx import ensemble, ow_ensemble, vr_ensemble, quantile_pinball
from ._approx_w import ensemble_w, ow_ensemble_w, vr_ensemble_w
from ._closed import (
    beta,
    binomial,
    exponential,
    exponentialM,
    twopexponential,
    gamma,
    gev,
    gpd,
    gtclogistic,
    gtcnormal,
    gtct,
    hypergeometric,
    laplace,
    logistic,
    loglaplace,
    loglogistic,
    lognormal,
    mixnorm,
    negbinom,
    normal,
    poisson,
    t,
    uniform,
)

try:
    from ._gufuncs import estimator_gufuncs, quantile_pinball_gufunc
    from ._gufuncs_w import estimator_gufuncs_w
except ImportError:
    estimator_gufuncs = None
    estimator_gufuncs_w = None
    quantile_pinball_gufunc = None

__all__ = [
    "ensemble",
    "ensemble_w",
    "ow_ensemble",
    "ow_ensemble_w",
    "vr_ensemble",
    "vr_ensemble_w",
    "beta",
    "binomial",
    "exponential",
    "exponentialM",
    "twopexponential",
    "gamma",
    "gev",
    "gpd",
    "gtclogistic",
    "gtcnormal",
    "gtct",
    "hypergeometric",
    "laplace",
    "logistic",
    "loglaplace",
    "loglogistic",
    "lognormal",
    "mixnorm",
    "negbinom",
    "normal",
    "poisson",
    "t",
    "uniform",
    "estimator_gufuncs",
    "estimator_gufuncs_w",
    "quantile_pinball",
    "quantile_pinball_gufunc",
]
