from abc import abstractmethod
from typing import TypeVar

from .arrayapi import Array

ArrayLike = TypeVar("ArrayLike", Array, float)


class AbstractBackend:
    """The abstract backend class."""

    backend_name: str

    @abstractmethod
    def crps_ensemble(
        self,
        fcts: Array,
        obs: ArrayLike,
        axis: int = -1,
        sorted_ensemble: bool = False,
        estimator: str = "pwm",
    ) -> Array:
        """Compute the CRPS for a finite ensemble."""

    @abstractmethod
    def crps_normal(self, mu: ArrayLike, sigma: ArrayLike, obs: ArrayLike) -> Array:
        """Compute the CRPS for the normal distribution."""

    @abstractmethod
    def crps_lognormal(
        self, mulog: ArrayLike, sigmalog: ArrayLike, obs: ArrayLike
    ) -> Array:
        """Compute the CRPS for the log normal distribution."""

    @abstractmethod
    def logscore_normal(
        self,
        mu: ArrayLike,
        sigma: ArrayLike,
        observation: ArrayLike,
    ) -> Array:
        """Compute the logarithmic score for the normal distribution."""

    @abstractmethod
    def energy_score(
        self, fcts: Array, obs: Array, m_axis: int = -2, v_axis: int = -1
    ) -> Array:
        """Compute the Energy Score for a multivariate finite ensemble."""

    @abstractmethod
    def variogram_score(
        self,
        fcts: Array,
        obs: Array,
        p: float = 1.0,
        m_axis: int = -2,
        v_axisc: int = -1,
    ) -> Array:
        """Compute the Variogram Score for a multivariate finite ensemble."""

    @abstractmethod
    def brier_score(self, fcts: ArrayLike, obs: ArrayLike) -> Array:
        """Compute the Brier Score for predicted probabilities."""
