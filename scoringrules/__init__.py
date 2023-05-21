from importlib.metadata import version

from scoringrules._brier import brier_score
from scoringrules._crps import crps_ensemble, crps_lognormal, crps_normal
from scoringrules._energy import energy_score
from scoringrules._variogram import variogram_score
from scoringrules.backend import register_backend

__version__ = version("scoringrules")


__all__ = [
    "register_backend",
    "crps_ensemble",
    "crps_normal",
    "crps_lognormal",
    "brier_score",
    "energy_score",
    "variogram_score",
]
