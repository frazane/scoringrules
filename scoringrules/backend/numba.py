from typing import TypeVar

import numpy as np

from .arrayapi import Array
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
    _variogram_score_gufunc,
)

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
    ) -> Array:
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
    def crps_normal(mu: ArrayLike, sigma: ArrayLike, observation: ArrayLike) -> Array:
        """Compute the CRPS for a normal distribution."""
        return _crps_normal_ufunc(mu, sigma, observation)

    @staticmethod
    def crps_lognormal(
        mulog: ArrayLike, sigmalog: ArrayLike, observation: ArrayLike
    ) -> Array:
        """Compute the CRPS for a log normal distribution."""
        return _crps_lognormal_ufunc(mulog, sigmalog, observation)

    @staticmethod
    def energy_score(
        forecasts: Array,
        observations: Array,
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Energy Score for a finite multivariate ensemble."""
        if m_axis != -2:
            forecasts = np.moveaxis(forecasts, m_axis, -2)
        if v_axis != -1:
            forecasts = np.moveaxis(forecasts, v_axis, -1)
            observations = np.moveaxis(observations, v_axis, -1)
        return _energy_score_gufunc(forecasts, observations)

    @staticmethod
    def brier_score(forecasts: ArrayLike, observations: ArrayLike) -> Array:
        """Compute the Brier Score for predicted probabilities."""
        return _brier_score_ufunc(forecasts, observations)

    @staticmethod
    def variogram_score(
        forecasts: Array,
        observations: Array,
        p: float = 1.0,
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Variogram Score for a finite multivariate ensemble."""
        return _variogram_score_gufunc(forecasts, observations, p)

    def __repr__(self) -> str:
        return "NumbaBackend"
