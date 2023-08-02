import typing as tp

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
    _crps_logistic_ufunc,
    _energy_score_gufunc,
    _logs_normal_ufunc,
    _owcrps_ensemble_nrg_gufunc,
    _variogram_score_gufunc,
)

ArrayLike = tp.TypeVar("ArrayLike", np.ndarray, float)
# ArrayLike = Array | float


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

        if not sorted_ensemble and estimator not in [
            "nrg",
            "akr",
            "akr_circperm",
            "fair",
        ]:
            if hasattr(forecasts, "__dask_graph__"):
                raise NotImplementedError(
                    "Estimators that need sorting of the ensemble do not yet work with dask. "
                    "Sort the ensemble beforehand or try another estimator (nrg, akr, akr_circperm or fair)"
                )
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
    def crps_logistic(mu: ArrayLike, sigma: ArrayLike, observation: ArrayLike) -> Array:
        """Compute the CRPS for a logistic distribution."""
        return _crps_logistic_ufunc(mu, sigma, observation)

    @staticmethod
    def owcrps_ensemble(
        forecasts: Array,
        observations: ArrayLike,
        w_func: tp.Callable= lambda x, *args: np.ones_like(x),
        w_funcargs: tuple=(),
    ) -> Array:
        """Compute the Threshold-Weighted CRPS for a finite ensemble."""
        fw = w_func(forecasts, *w_funcargs)
        ow = w_func(observations, *w_funcargs)
        return _owcrps_ensemble_nrg_gufunc(forecasts, observations, fw, ow)

    @staticmethod
    def twcrps_ensemble(
        forecasts: Array,
        observations: ArrayLike,
        v_func: tp.Callable= lambda x, *args: x,
        v_funcargs: tuple=(),
    ) -> Array:
        """Compute the Threshold-Weighted CRPS for a finite ensemble."""
        v_forecasts = v_func(forecasts, *v_funcargs)
        v_observations = v_func(observations, *v_funcargs)
        return _crps_ensemble_nrg_gufunc(v_forecasts, v_observations)

    @staticmethod
    def logs_normal(
        mu: ArrayLike,
        sigma: ArrayLike,
        observation: ArrayLike,
    ) -> Array:
        """Compute the logarithmic score for a normal distribution."""
        return _logs_normal_ufunc(mu, sigma, observation)

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
        m_axis: int = -1,
        v_axis: int = -2,
    ) -> Array:
        """Compute the Variogram Score for a finite multivariate ensemble."""
        if m_axis != -1:
            forecasts = np.moveaxis(forecasts, m_axis, -1)
            v_axis = v_axis - 1 if v_axis != 0 else v_axis
        if v_axis != -2:
            forecasts = np.moveaxis(forecasts, v_axis, -2)

        return _variogram_score_gufunc(forecasts, observations, p)

    def __repr__(self) -> str:
        return "NumbaBackend"
