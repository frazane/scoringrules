from importlib.metadata import version

from scoringrules._brier import brier_score
from scoringrules._crps import (
    crps_beta,
    crps_binomial,
    crps_ensemble,
    crps_exponential,
    crps_exponentialM,
    crps_gamma,
    crps_gev,
    crps_gpd,
    crps_hypergeometric,
    crps_laplace,
    crps_logistic,
    crps_loglaplace,
    crps_loglogistic,
    crps_lognormal,
    crps_normal,
    crps_uniform,
    crps_quantile,
    owcrps_ensemble,
    twcrps_ensemble,
    vrcrps_ensemble,
)
from scoringrules._energy import (
    energy_score,
    owenergy_score,
    twenergy_score,
    vrenergy_score,
)
from scoringrules._error_spread import error_spread_score
from scoringrules._interval import interval_score, weighted_interval_score
from scoringrules._logs import logs_normal
from scoringrules._variogram import (
    owvariogram_score,
    twvariogram_score,
    variogram_score,
    vrvariogram_score,
)
from scoringrules.backend import backends, register_backend

__version__ = version("scoringrules")


__all__ = [
    "register_backend",
    "backends",
    "crps_ensemble",
    "crps_beta",
    "crps_binomial",
    "crps_exponential",
    "crps_exponentialM",
    "crps_gamma",
    "crps_gev",
    "crps_gpd",
    "crps_hypergeometric",
    "crps_laplace",
    "crps_loglaplace",
    "crps_loglogistic",
    "crps_logistic",
    "crps_lognormal",
    "crps_normal",
    "crps_quantile",
    "crps_uniform",
    "owcrps_ensemble",
    "twcrps_ensemble",
    "vrcrps_ensemble",
    "logs_normal",
    "brier_score",
    "error_spread_score",
    "energy_score",
    "owenergy_score",
    "twenergy_score",
    "vrenergy_score",
    "variogram_score",
    "owvariogram_score",
    "twvariogram_score",
    "vrvariogram_score",
    "interval_score",
    "weighted_interval_score",
]
