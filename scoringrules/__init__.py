from importlib.metadata import version

from scoringrules._brier import brier_score
from scoringrules._crps import (
    crps_ensemble,
    crps_logistic,
    crps_lognormal,
    crps_normal,
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
    "crps_normal",
    "crps_lognormal",
    "crps_logistic",
    "owcrps_ensemble",
    "twcrps_ensemble",
    "vrcrps_ensemble",
    "logs_normal",
    "brier_score",
    "energy_score",
    "owenergy_score",
    "twenergy_score",
    "vrenergy_score",
    "variogram_score",
    "owvariogram_score",
    "twvariogram_score",
    "vrvariogram_score",
]
