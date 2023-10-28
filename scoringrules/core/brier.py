    


def brier_score(self, fcts: ArrayLike, obs: ArrayLike) -> Array:
        """Compute the Brier Score for predicted probabilities of events."""
        if self.np.any(fcts < 0.0) or self.np.any(fcts > 1.0 + EPSILON):
            raise ValueError("Forecasted probabilities must be within 0 and 1.")

        if not set(self.np.unique(obs)) <= {0, 1}:
            raise ValueError("Observations must be 0, 1, or NaN.")

        return self.np.asarray((fcts - obs) ** 2)