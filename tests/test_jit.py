"""Scores that must stay traceable under ``jax.jit``.

Known non-jittable, with the validation line that blocks each (the worklist for
a 1.0 ``validate=`` flag):

- weighted ``crps_ensemble`` (``ens_w=...``): ``xp.any(ens_w < 0)`` in
  ``core/utils_xp.py::univariate_weight_check``.
- ``crps_ensemble(..., nan_policy="raise")``: ``xp.any(nan_mask)`` in
  ``core/utils_xp.py::apply_nan_policy_ens_uv``.
- ``owcrps_ensemble`` / ``vrcrps_ensemble``: ``xp.any(obs_w < 0)`` in
  ``core/utils_xp.py::uv_weighted_score_weights``.
- ``crps_beta``: ``_is_scalar_value(lower, 0.0)`` (``.item()``) and
  ``xp.any(lower >= upper)`` in ``core/crps/_closed.py::beta``.
- ``crps_exponentialM``: ``_is_scalar_value`` (``.item()``) on ``location``,
  ``mass`` and ``scale`` in ``core/crps/_closed.py::exponentialM``.
- ``brier_score`` / ``log_score``: ``xp.any(fct < 0.0)`` in
  ``core/utils_xp.py::binary_array_check``.
- ``rps_score`` / ``rls_score``: ``xp.any(fct < 0.0)`` in
  ``core/utils_xp.py::categorical_array_check``.
- ``quantile_score``: ``xp.any(alpha < 0)`` in ``core/quantile.py::quantile_score``.
"""

import numpy as np
import pytest
import scoringrules as sr

jax = pytest.importorskip("jax")
jnp = jax.numpy

ESTIMATORS = ["nrg", "fair", "pwm", "int", "qd", "akr", "akr_circperm"]


def test_crps_normal_jit():
    obs = jnp.asarray(np.random.randn(5))
    mu = jnp.asarray(np.random.randn(5))
    sigma = jnp.asarray(np.random.uniform(0.5, 2.0, 5))
    res = jax.jit(sr.crps_normal)(obs, mu, sigma)
    np.testing.assert_allclose(res, sr.crps_normal(obs, mu, sigma), rtol=1e-6)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_crps_ensemble_jit(estimator):
    obs = jnp.asarray(np.random.randn(5))
    fct = jnp.asarray(np.random.randn(5, 11))

    @jax.jit
    def f(o, x):
        return sr.crps_ensemble(o, x, estimator=estimator)

    expected = sr.crps_ensemble(obs, fct, estimator=estimator)
    np.testing.assert_allclose(f(obs, fct), expected, rtol=1e-5, atol=1e-7)


def test_interval_scores_jit():
    rng = np.random.default_rng(0)  # float32 rounding: keep the draw fixed
    obs = jnp.asarray(rng.normal(size=5))
    alpha = jnp.asarray([0.2, 0.5, 0.8])
    half = jnp.asarray(rng.uniform(0.1, 2.0, (5, 3)))
    lower, upper = obs[:, None] - half, obs[:, None] + half
    np.testing.assert_allclose(
        jax.jit(sr.interval_score)(obs, lower, upper, alpha),
        sr.interval_score(obs, lower, upper, alpha),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jax.jit(sr.weighted_interval_score)(obs, obs, lower, upper, alpha),
        sr.weighted_interval_score(obs, obs, lower, upper, alpha),
        rtol=1e-6,
        atol=1e-6,
    )


def test_error_spread_score_jit():
    # seeded: the score divides by the ensemble variance, so float32 rounding
    # differences between the jitted and eager runs can exceed a tight rtol
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.normal(size=5))
    fct = jnp.asarray(rng.normal(size=(5, 11)))
    np.testing.assert_allclose(
        jax.jit(sr.error_spread_score)(obs, fct),
        sr.error_spread_score(obs, fct),
        rtol=1e-5,
        atol=1e-5,
    )
