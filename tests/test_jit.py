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
