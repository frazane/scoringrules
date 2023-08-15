from types import ModuleType

import numpy as np
import typing as tp

from .arrayapi import Array, ArrayAPIProtocol
from .base import AbstractBackend

INV_SQRT_PI = 1 / np.sqrt(np.pi)
EPSILON = 1e-6

ArrayLike = tp.TypeVar("ArrayLike", Array, float)


class ArrayAPIBackend(AbstractBackend):
    """A class for python array-API-standard compatible backends."""

    _supported_backends = ["numpy", "jax"]
    np: ArrayAPIProtocol
    scipy: ModuleType

    def __init__(self, backend_name: str):
        if backend_name == "numpy":
            import numpy as np
            import scipy

            self.scipy = scipy  # type: ignore
            self.np = np  # type: ignore
        elif backend_name == "jax":
            import jax.numpy as jnp
            from jax import scipy

            self.scipy = scipy  # type: ignore
            self.np = jnp  # type: ignore
        else:
            raise ValueError(
                f"{backend_name} is not a supported array-API backend. "
                f"Supported backends are {self._supported_backends}"
            )

        self.backend_name = backend_name

    def crps_ensemble(
        self,
        fcts: Array,
        obs: ArrayLike,
        axis: int = -1,
        sorted_ensemble: bool = False,
        estimator: str = "pwm",
    ) -> Array:
        """Compute the CRPS for a finite ensemble."""
        obs: Array = self.np.asarray(obs)

        if axis != -1:
            fcts = self.np.moveaxis(fcts, axis, -1)

        if not sorted_ensemble and estimator != "nrg":
            fcts = self.np.sort(fcts, axis=-1)

        if estimator == "nrg":
            out = self._crps_ensemble_nrg(fcts, obs)
        elif estimator == "pwm":
            out = self._crps_ensemble_pwm(fcts, obs)
        elif estimator == "fair":
            out = self._crps_ensemble_fair(fcts, obs)
        else:
            raise ValueError(
                f"{estimator} is not a valid estimator for numpy-API backend."
            )

        return out

    def crps_normal(self, mu: ArrayLike, sigma: ArrayLike, obs: ArrayLike) -> Array:
        """Compute the CRPS for the normal distribution."""
        ω = (obs - mu) / sigma
        return sigma * (
            ω * (2 * self._norm_cdf(ω) - 1) + 2 * self._norm_pdf(ω) - INV_SQRT_PI
        )

    def crps_lognormal(
        self, mulog: ArrayLike, sigmalog: ArrayLike, obs: ArrayLike
    ) -> Array:
        """Compute the CRPS for the lognormal distribution."""
        ω = (self.np.log(obs) - mulog) / sigmalog
        ex = 2 * self.np.exp(mulog + sigmalog**2 / 2)
        return obs * (2 * self._norm_cdf(ω) - 1) - ex * (
            self._norm_cdf(ω - sigmalog)
            + self._norm_cdf(sigmalog / self.np.sqrt(2))
            - 1
        )

    def crps_logistic(self, mu: ArrayLike, sigma: ArrayLike, obs: ArrayLike) -> Array:
        """Compute the CRPS for the normal distribution."""
        ω = (obs - mu) / sigma
        return sigma * (ω - 2 * self.np.log(self._logis_cdf(ω)) - 1)

    def owcrps_ensemble(
        self,
        fcts: Array,
        obs: ArrayLike,
        w_func: tp.Callable = lambda x, *args: np.ones_like(x),
        w_funcargs: tuple = (),
        axis: int = -1,
    ) -> Array:
        """Compute the Outcome-Weighted CRPS for a finite ensemble."""
        obs: Array = self.np.asarray(obs)

        if axis != -1:
            fcts = self.np.moveaxis(fcts, axis, -1)

        fw = w_func(fcts, *w_funcargs)
        ow = w_func(obs, *w_funcargs)

        out = self._owcrps_ensemble_nrg(fcts, obs, fw, ow)

        return out
    
    def twcrps_ensemble(
        self,
        fcts: Array,
        obs: ArrayLike,
        v_func: tp.Callable = lambda x, *args: x,
        v_funcargs: tuple = (),
        axis: int = -1,
    ) -> Array:
        """Compute the Threshold-Weighted CRPS for a finite ensemble."""
        obs: Array = self.np.asarray(obs)

        if axis != -1:
            fcts = self.np.moveaxis(fcts, axis, -1)
            
        v_fcts = v_func(fcts, *v_funcargs)
        v_obs = v_func(obs, *v_funcargs)

        out = self._crps_ensemble_nrg(v_fcts, v_obs)

        return out
    
    def vrcrps_ensemble(
        self,
        fcts: Array,
        obs: ArrayLike,
        w_func: tp.Callable = lambda x, *args: np.ones_like(x),
        w_funcargs: tuple = (),
        axis: int = -1,
    ) -> Array:
        """Compute the Vertically Re-scaled CRPS for a finite ensemble."""
        obs: Array = self.np.asarray(obs)

        if axis != -1:
            fcts = self.np.moveaxis(fcts, axis, -1)
            
        fw = w_func(fcts, *w_funcargs)
        ow = w_func(obs, *w_funcargs)

        out = self._vrcrps_ensemble_nrg(fcts, obs, fw, ow)

        return out
    
    def energy_score(self, fcts: Array, obs: Array, m_axis=-2, v_axis=-1) -> Array:
        """
        Compute the energy score based on a finite ensemble.

        Note: there's a dirty trick to avoid computing the norm on a vector that contains
        zeros, because in JAX the gradient of the norm of zero returns nans.
        """
        M: int = fcts.shape[m_axis]
        err_norm = self.np.linalg.norm(
            fcts - self.np.expand_dims(obs, axis=m_axis), axis=v_axis, keepdims=True
        )
        E_1 = self.np.sum(err_norm, axis=m_axis) / M

        ens_diff = self.np.expand_dims(fcts, axis=m_axis) - self.np.expand_dims(
            fcts, axis=m_axis - 1
        )
        ens_diff = self.np.where(ens_diff == 0.0, self.np.ones_like(ens_diff), ens_diff)
        spread_norm = self.np.linalg.norm(ens_diff, axis=v_axis, keepdims=True)
        spread_norm = self.np.where(
            self.np.eye(M, M, dtype=bool), self.np.zeros_like(spread_norm), spread_norm
        )
        E_2 = self.np.sum(spread_norm, axis=(m_axis, m_axis - 1), keepdims=True) / (
            M * (M - 1)
        )
        return self.np.squeeze(E_1 - 0.5 * E_2)
    
    def owenergy_score(
        self,
        fcts: Array,
        obs: Array,
        w_func: tp.Callable = lambda x, *args: 1.0,
        w_funcargs: tuple = (),
        m_axis=-2, 
        v_axis=-1,
    ) -> Array:
        """Compute the outcome-weighted energy score based on a finite ensemble."""
        M: int = fcts.shape[m_axis]

        w_func_wrap = lambda x: w_func(x, *w_funcargs)
        fw = self.np.apply_along_axis(w_func_wrap, axis=-1, arr=fcts)
        ow = self.np.apply_along_axis(w_func_wrap, axis=0, arr=obs)
        wbar = self.np.mean(fw)

        err_norm = self.np.linalg.norm(
            fcts - self.np.expand_dims(obs, axis=m_axis), axis=v_axis, keepdims=True
        )
        E_1 = self.np.sum(err_norm * self.np.expand_dims(fw, axis=v_axis) * ow, axis=m_axis) / (M * wbar)

        ens_diff = self.np.expand_dims(fcts, axis=m_axis) - self.np.expand_dims(
            fcts, axis=m_axis - 1
        )
        spread_norm = self.np.linalg.norm(ens_diff, axis=v_axis, keepdims=True)
        
        fw_diff = self.np.expand_dims(fw, axis=m_axis)*self.np.expand_dims(fw, axis=v_axis)
        E_2 = self.np.sum(spread_norm * self.np.expand_dims(fw_diff, axis=-1) * ow,
                          axis=(m_axis, m_axis - 1),
                          keepdims=True) / (M**2 + wbar**2)
        return self.np.squeeze(E_1 - 0.5 * E_2)
    
    def twenergy_score(
        self,
        fcts: Array,
        obs: Array,
        v_func: tp.Callable = lambda x, *args: x,
        v_funcargs: tuple = (),
        m_axis=-2,
        v_axis=-1
    ) -> Array:
        """
        Compute the threshold-weighted energy score based on a finite ensemble.
        """
        if m_axis != -2:
            fcts = self.np.moveaxis(fcts, m_axis, -2)
            v_axis = v_axis - 1 if v_axis != 0 else v_axis
        if v_axis != -1:
            fcts = self.np.moveaxis(fcts, v_axis, -1)
            obs = self.np.moveaxis(obs, v_axis, -1)
        
        v_fcts = v_func(fcts, *v_funcargs)
        v_obs = v_func(obs, *v_funcargs)
        return self.energy_score(v_fcts, v_obs)
    
    def vrenergy_score(
        self,
        fcts: Array,
        obs: Array,
        w_func: tp.Callable = lambda x, *args: 1.0,
        w_funcargs: tuple = (),
        m_axis=-2, 
        v_axis=-1,
    ) -> Array:
        """Compute the vertically re-scaled energy score based on a finite ensemble."""
        M: int = fcts.shape[m_axis]

        w_func_wrap = lambda x: w_func(x, *w_funcargs)
        fw = self.np.apply_along_axis(w_func_wrap, axis=-1, arr=fcts)
        ow = self.np.apply_along_axis(w_func_wrap, axis=0, arr=obs)

        err_norm = self.np.linalg.norm(
            fcts - self.np.expand_dims(obs, axis=m_axis), axis=v_axis, keepdims=True
        )
        E_1 = self.np.sum(err_norm * self.np.expand_dims(fw, axis=v_axis) * ow, axis=m_axis) / M

        ens_diff = self.np.expand_dims(fcts, axis=m_axis) - self.np.expand_dims(
            fcts, axis=m_axis - 1
        )
        spread_norm = self.np.linalg.norm(ens_diff, axis=v_axis, keepdims=True)
        
        fw_diff = self.np.expand_dims(fw, axis=m_axis)*self.np.expand_dims(fw, axis=v_axis)
        E_2 = self.np.sum(spread_norm * self.np.expand_dims(fw_diff, axis=-1),
                          axis=(m_axis, m_axis - 1),
                          keepdims=True) / (M ** 2)
        
        wbar = self.np.mean(fw)
        rhobar = self.np.mean(self.np.linalg.norm(fcts, axis=v_axis) * fw)
        E_3 = (rhobar - self.np.linalg.norm(obs) * ow) * (wbar - ow)
        return self.np.squeeze(E_1 - 0.5 * E_2 + E_3)

    def brier_score(self, fcts: ArrayLike, obs: ArrayLike) -> Array:
        """Compute the Brier Score for predicted probabilities of events."""
        if self.np.any(fcts < 0.0) or self.np.any(fcts > 1.0 + EPSILON):
            raise ValueError("Forecasted probabilities must be within 0 and 1.")

        if not set(self.np.unique(obs)) <= {0, 1}:
            raise ValueError("Observations must be 0, 1, or NaN.")

        return self.np.asarray((fcts - obs) ** 2)

    def variogram_score(
        self,
        forecasts: Array,
        observations: Array,
        p: float = 1,
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Variogram Score for a multivariate finite ensemble."""
        if m_axis != -2:
            forecasts = self.np.moveaxis(forecasts, m_axis, -2)
            v_axis = v_axis - 1 if v_axis != 0 else v_axis
        if v_axis != -1:
            forecasts = self.np.moveaxis(forecasts, v_axis, -1)
            observations = self.np.moveaxis(observations, v_axis, -1)

        M: int = forecasts.shape[-1]
        fct_diff = (
            self.np.abs(
                self.np.expand_dims(forecasts, -2) - self.np.expand_dims(forecasts, -3)
            )
            ** p
        )
        vfcts = self.np.sum(fct_diff, axis=-1) / M
        obs_diff = (
            self.np.abs(observations[..., None] - observations[..., None, :]) ** p
        )
        out = self.np.sum((obs_diff - vfcts) ** 2, axis=(-2, -1))
        return out
    
    def twvariogram_score(
        self,
        forecasts: Array,
        observations: Array,
        v_func: tp.Callable = lambda x, *args: x,
        v_funcargs: tuple = (),
        p: float = 1,
        m_axis: int = -2,
        v_axis: int = -1,
    ) -> Array:
        """Compute the Threshold-Weighted Variogram Score for a multivariate finite ensemble."""
        if m_axis != -2:
            forecasts = self.np.moveaxis(forecasts, m_axis, -2)
            v_axis = v_axis - 1 if v_axis != 0 else v_axis
        if v_axis != -1:
            forecasts = self.np.moveaxis(forecasts, v_axis, -1)
            observations = self.np.moveaxis(observations, v_axis, -1)

        v_fcts = v_func(forecasts, *v_funcargs)
        v_obs = v_func(observations, *v_funcargs)
        return self.variogram_score(v_fcts, v_obs, p)

    def logs_normal(
        self,
        mu: ArrayLike,
        sigma: ArrayLike,
        obs: ArrayLike,
        negative: bool = True,
    ):
        """Compute the logarithmic score for the normal distribution."""
        constant = -1.0 if negative else 1.0
        ω = (obs - mu) / sigma
        prob = self._norm_pdf(ω) / sigma
        return constant * self.np.log(prob)

    def _crps_ensemble_fair(self, fcts: Array, obs: Array) -> Array:
        """Fair version of the CRPS estimator based on the energy form."""
        M: int = fcts.shape[-1]
        e_1 = self.np.sum(self.np.abs(obs[..., None] - fcts), axis=-1) / M
        e_2 = self.np.sum(
            self.np.abs(fcts[..., None] - fcts[..., None, :]),
            axis=(-1, -2),
        ) / (M * (M - 1))
        return e_1 - 0.5 * e_2

    def _crps_ensemble_nrg(self, fcts: Array, obs: Array) -> Array:
        """CRPS estimator based on the energy form."""
        M: int = fcts.shape[-1]
        e_1 = self.np.sum(self.np.abs(obs[..., None] - fcts), axis=-1) / M
        e_2 = self.np.sum(
            self.np.abs(
                self.np.expand_dims(fcts, axis=-1) - self.np.expand_dims(fcts, axis=-2)
            ),
            axis=(-1, -2),
        ) / (M**2)
        return e_1 - 0.5 * e_2

    def _crps_ensemble_pwm(self, fcts: Array, obs: Array) -> Array:
        """CRPS estimator based on the probability weighted moment (PWM) form."""
        M: int = fcts.shape[-1]
        expected_diff = self.np.sum(self.np.abs(obs[..., None] - fcts), axis=-1) / M
        β_0 = self.np.sum(fcts, axis=-1) / M
        β_1 = self.np.sum(fcts * self.np.arange(M), axis=-1) / (M * (M - 1))
        return expected_diff + β_0 - 2.0 * β_1

    def _owcrps_ensemble_nrg(self, fcts: Array, obs: Array, fw: Array, ow: Array) -> Array:
        """Outcome-Weighted CRPS estimator based on the energy form."""
        M: int = fcts.shape[-1]
        wbar = self.np.mean(fw, axis=1)
        e_1 = self.np.sum(self.np.abs(obs[..., None] - fcts) * fw, axis=-1) * ow / (M * wbar)
        e_2 = self.np.sum(
            self.np.abs(
                self.np.expand_dims(fcts * fw, axis=-1) - self.np.expand_dims(fcts * fw, axis=-2)
            ),
            axis=(-1, -2),
        ) 
        e_2 *= ow / (M**2 * wbar**2)
        return e_1 - 0.5 * e_2
    
    def _vrcrps_ensemble_nrg(self, fcts: Array, obs: Array, fw: Array, ow: Array) -> Array:
        """Vertically Re-scaled CRPS estimator based on the energy form."""
        M: int = fcts.shape[-1]
        e_1 = self.np.sum(self.np.abs(obs[..., None] - fcts) * fw, axis=-1) * ow / M
        e_2 = self.np.sum(
            self.np.abs(
                self.np.expand_dims(fcts * fw, axis=-1) - self.np.expand_dims(fcts * fw, axis=-2)
            ),
            axis=(-1, -2),
        ) / (M**2)
        e_3 = (self.np.mean(self.np.abs(fcts) * fw, axis=-1) - obs[..., None] * ow[..., None])
        e_3 *= (self.np.mean(fw, axis=1) - ow)
        return e_1 - 0.5 * e_2 + e_3

    def _norm_cdf(self, x: ArrayLike) -> Array:
        """Cumulative distribution function for the standard normal distribution."""
        return (1.0 + self.scipy.special.erf(x / self.np.sqrt(2.0))) / 2.0

    def _norm_pdf(self, x: ArrayLike) -> Array:
        """Probability density function for the standard normal distribution."""
        return (1.0 / self.np.sqrt(2.0 * self.np.pi)) * self.np.exp(-(x**2) / 2)

    def _logis_cdf(self, x: ArrayLike) -> Array:
        """Cumulative distribution function for the standard logistic distribution."""
        return (1 / (1 + np.exp(-x)))

    def __repr__(self) -> str:
        return f"NumpyAPIBackend('{self.backend_name}')"
