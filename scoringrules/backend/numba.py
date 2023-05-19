from typing import TypeVar

import numpy as np

from .base import AbstractBackend
from .gufuncs import (
    _brier_score_ufunc,
    _crps_ensemble_akr_circperm_gufunc,
    _crps_ensemble_akr_gufunc,
    _crps_ensemble_fair_gufunc,
    _crps_ensemble_int_gufunc,
    _crps_ensemble_nrg_gufunc,
    _crps_ensemble_pwm_gufunc,
    _crps_ensemble_qd_gufunc,
    _crps_lognormal_ufunc,
    _crps_normal_ufunc,
    _energy_score_gufunc,
)

Array = np.ndarray
ArrayLike = TypeVar("ArrayLike", np.ndarray, float)


class NumbaBackend(AbstractBackend):
    """The numba backend."""

    def __init__(self):
        self.backend_name = "numba"

    @staticmethod
    def crps_ensemble(
        forecasts: Array,
        observations: ArrayLike,
        axis=-1,
        sorted_ensemble=False,
        estimator="int",
    ) -> ArrayLike:
        """Compute the CRPS for a finite ensemble."""
        if axis != -1:
            forecasts = np.moveaxis(forecasts, axis, -1)

        if not sorted_ensemble and estimator != "nrg":
            forecasts = np.sort(forecasts, axis=-1)

        if estimator == "int":
            out = _crps_ensemble_int_gufunc(forecasts, observations)
        elif estimator == "qd":
            out = _crps_ensemble_qd_gufunc(forecasts, observations)
        elif estimator == "nrg":
            out = _crps_ensemble_nrg_gufunc(forecasts, observations)
        elif estimator == "pwm":
            out = _crps_ensemble_pwm_gufunc(forecasts, observations)
        elif estimator == "fair":
            out = _crps_ensemble_fair_gufunc(forecasts, observations)
        elif estimator == "akr":
            out = _crps_ensemble_akr_gufunc(forecasts, observations)
        elif estimator == "akr_circperm":
            out = _crps_ensemble_akr_circperm_gufunc(forecasts, observations)
        else:
            raise ValueError(f"{estimator} is not a valid estimator for numba backend.")

        return out

    @staticmethod
    def crps_normal(
        mu: ArrayLike, sigma: ArrayLike, observation: ArrayLike
    ) -> ArrayLike:
        """Compute the CRPS for a normal distribution."""
        return _crps_normal_ufunc(mu, sigma, observation)

    @staticmethod
    def crps_lognormal(
        mulog: ArrayLike, sigmalog: ArrayLike, observation: ArrayLike
    ) -> ArrayLike:
        """Compute the CRPS for a log normal distribution."""
        return _crps_lognormal_ufunc(mulog, sigmalog, observation)

    @staticmethod
    def energy_score(forecasts: Array, observations: Array) -> ArrayLike:
        """Compute the Energy Score for a finite multivariate ensemble."""
        return _energy_score_gufunc(forecasts, observations)

    @staticmethod
    def brier_score(forecasts: ArrayLike, observations: ArrayLike) -> ArrayLike:
        """Compute the Brier Score for a finite multivariate ensemble."""
        return _brier_score_ufunc(forecasts, observations)

    def __repr__(self):
        return "NumbaBackend"
