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
    _owcrps_ensemble_nrg_gufunc,
    _vrcrps_ensemble_nrg_gufunc,
    _energy_score_gufunc,
    _owenergy_score_gufunc,
    _vrenergy_score_gufunc,
    _logs_normal_ufunc,
    _variogram_score_gufunc,
    _owvariogram_score_gufunc,
    _vrvariogram_score_gufunc,
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
        w_func: tp.Callable = lambda x, *args: np.ones_like(x),
        w_funcargs: tuple = (),
        axis=-1,
    ) -> Array:
        """Compute the Outcome-Weighted CRPS for a finite ensemble."""
        if axis != -1:
            forecasts = np.moveaxis(forecasts, axis, -1)
            
        fw = w_func(forecasts, *w_funcargs)
        ow = w_func(observations, *w_funcargs)
        return _owcrps_ensemble_nrg_gufunc(forecasts, observations, fw, ow)

    @staticmethod
    def twcrps_ensemble(
        forecasts: Array,
        observations: ArrayLike,
        v_func: tp.Callable = lambda x, *args: x,
        v_funcargs: tuple = (),
        axis=-1,
    ) -> Array:
        """Compute the Threshold-Weighted CRPS for a finite ensemble."""
        if axis != -1:
            forecasts = np.moveaxis(forecasts, axis, -1)
            
        v_forecasts = v_func(forecasts, *v_funcargs)
        v_observations = v_func(observations, *v_funcargs)
        return _crps_ensemble_nrg_gufunc(v_forecasts, v_observations)

    @staticmethod
    def vrcrps_ensemble(
        forecasts: Array,
        observations: ArrayLike,
        w_func: tp.Callable = lambda x, *args: np.ones_like(x),
        w_funcargs: tuple = (),
        axis=-1,
    ) -> Array:
        """Compute the Vertically Re-scaled CRPS for a finite ensemble."""
        if axis != -1:
            forecasts = np.moveaxis(forecasts, axis, -1)
            
        fw = w_func(forecasts, *w_funcargs)
        ow = w_func(observations, *w_funcargs)
        return _vrcrps_ensemble_nrg_gufunc(forecasts, observations, fw, ow)

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
    def owenergy_score(
        forecasts: Array,
        observations: Array,
        w_func: tp.Callable = lambda x, *args: 1.0,
        w_funcargs: tuple = (),
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Outcome-Weighted Energy Score for a finite multivariate ensemble."""
        if m_axis != -2:
            forecasts = np.moveaxis(forecasts, m_axis, -2)
        if v_axis != -1:
            forecasts = np.moveaxis(forecasts, v_axis, -1)
            observations = np.moveaxis(observations, v_axis, -1)

        w_func_wrap = lambda x: w_func(x, *w_funcargs)
        fw = np.apply_along_axis(w_func_wrap, axis=-1, arr=forecasts)
        ow = np.apply_along_axis(w_func_wrap, axis=0, arr=observations)
        return _owenergy_score_gufunc(forecasts, observations, fw, ow)

    @staticmethod
    def twenergy_score(
        forecasts: Array,
        observations: Array,
        v_func: tp.Callable = lambda x, *args: x,
        v_funcargs: tuple = (),
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Threshold-Weighted Energy Score for a finite multivariate ensemble."""
        if m_axis != -2:
            forecasts = np.moveaxis(forecasts, m_axis, -2)
        if v_axis != -1:
            forecasts = np.moveaxis(forecasts, v_axis, -1)
            observations = np.moveaxis(observations, v_axis, -1)
            
        v_forecasts = v_func(forecasts, *v_funcargs)
        v_observations = v_func(observations, *v_funcargs)
        return _energy_score_gufunc(v_forecasts, v_observations)
    
    @staticmethod
    def vrenergy_score(
        forecasts: Array,
        observations: Array,
        w_func: tp.Callable = lambda x, *args: 1.0,
        w_funcargs: tuple = (),
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Vertically Re-scaled Energy Score for a finite multivariate ensemble."""
        if m_axis != -2:
            forecasts = np.moveaxis(forecasts, m_axis, -2)
        if v_axis != -1:
            forecasts = np.moveaxis(forecasts, v_axis, -1)
            observations = np.moveaxis(observations, v_axis, -1)
            
        w_func_wrap = lambda x: w_func(x, *w_funcargs)
        fw = np.apply_along_axis(w_func_wrap, axis=-1, arr=forecasts)
        ow = np.apply_along_axis(w_func_wrap, axis=0, arr=observations)
        
        return _vrenergy_score_gufunc(forecasts, observations, fw, ow)

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
        if m_axis != -2:
            forecasts = np.moveaxis(forecasts, m_axis, -2)
        if v_axis != -1:
            forecasts = np.moveaxis(forecasts, v_axis, -1)
            observations = np.moveaxis(observations, v_axis, -1)

        return _variogram_score_gufunc(forecasts, observations, p)
    
    @staticmethod
    def owvariogram_score(
        forecasts: Array,
        observations: Array,
        p: float = 1.0,
        w_func: tp.Callable = lambda x, *args: 1.0,
        w_funcargs: tuple = (),
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Outcome-Weighted Variogram Score for a finite multivariate ensemble."""
        if m_axis != -2:
            forecasts = np.moveaxis(forecasts, m_axis, -2)
        if v_axis != -1:
            forecasts = np.moveaxis(forecasts, v_axis, -1)
            observations = np.moveaxis(observations, v_axis, -1)

        w_func_wrap = lambda x: w_func(x, *w_funcargs)
        fw = np.apply_along_axis(w_func_wrap, axis=-1, arr=forecasts)
        ow = np.apply_along_axis(w_func_wrap, axis=0, arr=observations)
        return _owvariogram_score_gufunc(forecasts, observations, p, fw, ow)
    
    @staticmethod
    def twvariogram_score(
        forecasts: Array,
        observations: Array,
        p: float = 1.0,
        v_func: tp.Callable = lambda x, *args: x,
        v_funcargs: tuple = (),
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Threshold-Weighted Variogram Score for a finite multivariate ensemble."""
        if m_axis != -2:
            forecasts = np.moveaxis(forecasts, m_axis, -2)
        if v_axis != -1:
            forecasts = np.moveaxis(forecasts, v_axis, -1)
            observations = np.moveaxis(observations, v_axis, -1)

        v_forecasts = v_func(forecasts, *v_funcargs)
        v_observations = v_func(observations, *v_funcargs)
        return _variogram_score_gufunc(v_forecasts, v_observations, p)
    
    @staticmethod
    def vrvariogram_score(
        forecasts: Array,
        observations: Array,
        p: float = 1.0,
        w_func: tp.Callable = lambda x, *args: 1.0,
        w_funcargs: tuple = (),
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Vertically Re-scaled Variogram Score for a finite multivariate ensemble."""
        if m_axis != -2:
            forecasts = np.moveaxis(forecasts, m_axis, -2)
        if v_axis != -1:
            forecasts = np.moveaxis(forecasts, v_axis, -1)
            observations = np.moveaxis(observations, v_axis, -1)

        w_func_wrap = lambda x: w_func(x, *w_funcargs)
        fw = np.apply_along_axis(w_func_wrap, axis=-1, arr=forecasts)
        ow = np.apply_along_axis(w_func_wrap, axis=0, arr=observations)
        return _vrvariogram_score_gufunc(forecasts, observations, p, fw, ow)

    def __repr__(self) -> str:
        return "NumbaBackend"
