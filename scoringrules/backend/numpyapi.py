from types import ModuleType
from typing import Any, TypeVar

import numpy as np

from .base import AbstractBackend

INV_SQRT_PI = 1 / np.sqrt(np.pi)
EPSILON = 1e-6

Array = TypeVar("Array", np.ndarray, Any)
ArrayLike = TypeVar("ArrayLike", np.ndarray, Any, float)


class NumpyAPIBackend(AbstractBackend):
    """A class for numpy-API compatible backends."""

    np: ModuleType
    special: ModuleType

    def __init__(self, backend_name: str):
        if backend_name == "numpy":
            import numpy as np
            from scipy import special

            self.special = special
            self.np = np
        if backend_name == "jax":
            import jax.numpy as jnp
            from jax.scipy import special

            self.special = special
            self.np = jnp

        self.backend_name = backend_name

    def crps_ensemble(
        self,
        fcts: Array,
        obs: ArrayLike,
        axis=-1,
        sorted_ensemble=False,
        estimator="pwm",
    ) -> ArrayLike:
        """Compute the CRPS for a finite ensemble."""
        if axis != -1:
            fcts = np.moveaxis(fcts, axis, -1)

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

    def crps_normal(self, mu: ArrayLike, sigma: ArrayLike, obs: ArrayLike) -> ArrayLike:
        """Compute the CRPS for the normal distribution."""
        ω = (obs - mu) / sigma
        return sigma * (
            ω * (2 * self._norm_cdf(ω) - 1) + 2 * self._norm_pdf(ω) - INV_SQRT_PI
        )

    def crps_lognormal(
        self, mulog: ArrayLike, sigmalog: ArrayLike, obs: ArrayLike
    ) -> ArrayLike:
        """Compute the CRPS for the lognormal distribution."""
        ω = (self.np.log(obs) - mulog) / sigmalog
        ex = 2 * self.np.exp(mulog + sigmalog**2 / 2)
        return obs * (2 * self._norm_cdf(ω) - 1) - ex * (
            self._norm_cdf(ω - sigmalog)
            + self._norm_cdf(sigmalog / self.np.sqrt(2))
            - 1
        )

    def energy_score(self, fcts: Array, obs: Array, m_axis=-2, v_axis=-1) -> ArrayLike:
        """
        Compute the energy score based on a finite ensemble.

        Note: there's a dirty trick to avoid computing the norm on a vector that contains
        zeros, because in JAX the gradient of the norm of zero returns nans.
        """
        M = fcts.shape[m_axis]
        err_norm = self.np.linalg.norm(
            fcts - self.np.expand_dims(obs, v_axis), axis=v_axis, keepdims=True
        )
        E_1 = self.np.sum(err_norm, axis=m_axis) / M

        ens_diff = self.np.expand_dims(fcts, m_axis) - self.np.expand_dims(
            fcts, m_axis - 1
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

    def brier_score(self, fcts: ArrayLike, obs: ArrayLike) -> ArrayLike:
        """Compute the Brier Score for predicted probabilities of events."""
        if self.np.any(fcts < 0.0) or self.np.any(fcts > 1.0 + EPSILON):
            raise ValueError("Forecasted probabilities must be within 0 and 1.")

        if not set(self.np.unique(obs)) <= {0, 1}:
            raise ValueError("Observations must be 0, 1, or NaN.")

        return (fcts - obs) ** 2

    def _crps_ensemble_fair(self, fcts, obs):
        """Fair version of the CRPS estimator based on the energy form."""
        M = fcts.shape[-1]
        e_1 = self.np.nansum(self.np.abs(obs[..., None] - fcts), axis=-1) / M
        e_2 = self.np.nansum(
            self.np.abs(fcts[..., None] - fcts[..., None, :]),
            axis=(-1, -2),
        ) / (M * (M - 1))
        return e_1 - 0.5 * e_2

    def _crps_ensemble_nrg(self, fcts, obs):
        """CRPS estimator based on the energy form."""
        M = fcts.shape[-1]
        e_1 = self.np.nansum(self.np.abs(obs[..., None] - fcts), axis=-1) / M
        e_2 = self.np.nansum(
            self.np.abs(self.np.expand_dims(fcts, -1) - self.np.expand_dims(fcts, -2)),
            axis=(-1, -2),
        ) / (M**2)
        return e_1 - 0.5 * e_2

    def _crps_ensemble_pwm(self, fcts, obs):
        """CRPS estimator based on the probability weighted moment (PWM) form."""
        M = fcts.shape[-1]
        expected_diff = self.np.sum(self.np.abs(obs[..., None] - fcts), axis=-1) / M
        β_0 = self.np.sum(fcts, axis=-1) / M
        β_1 = self.np.sum(fcts * self.np.arange(M), axis=-1) / (M * (M - 1))
        return expected_diff + β_0 - 2 * β_1

    def _norm_cdf(self, x):
        """Cumulative distribution function for the standard normal distribution."""
        return (1.0 + self.special.erf(x / self.np.sqrt(2.0))) / 2.0

    def _norm_pdf(self, x):
        """Probability density function for the standard normal distribution."""
        return (1 / self.np.sqrt(2 * self.np.pi)) * self.np.exp(-(x**2) / 2)

    def __repr__(self):
        return f"NumpyAPIBackend('{self.backend_name}')"
