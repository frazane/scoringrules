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
- weighted ``es_ensemble`` / ``twes_ensemble`` / ``vs_ensemble`` / ``twvs_ensemble``
  (``ens_w=...``, and ``es_ensemble(..., nan_policy="omit")``): ``xp.any(ens_w < 0)``
  in ``core/utils_xp.py::multivariate_weight_check``.
- ``es_ensemble(..., nan_policy="raise")``: ``xp.any(nan_mask)`` in
  ``core/utils_xp.py::apply_nan_policy_ens_mv``.
- ``owes_ensemble`` / ``vres_ensemble`` / ``owvs_ensemble`` / ``vrvs_ensemble``:
  ``xp.any(obs_w < 0)`` in ``core/utils_xp.py::mv_weighted_score_weights``.
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


@pytest.mark.parametrize("bias", [False, True])
def test_dss_ensemble_jit(bias):
    rng = np.random.default_rng(0)  # float32 rounding: keep the draw fixed
    obs = jnp.asarray(rng.normal(size=(5, 3)))
    fct = jnp.asarray(rng.normal(size=(5, 11, 3)))

    @jax.jit
    def uv(o, x):
        return sr.dssuv_ensemble(o, x, bias=bias)

    @jax.jit
    def mv(o, x):
        return sr.dssmv_ensemble(o, x, bias=bias)

    expected = sr.dssuv_ensemble(obs[:, 0], fct[..., 0], bias=bias)
    np.testing.assert_allclose(
        uv(obs[:, 0], fct[..., 0]), expected, rtol=1e-5, atol=1e-5
    )
    expected = sr.dssmv_ensemble(obs, fct, bias=bias)
    np.testing.assert_allclose(mv(obs, fct), expected, rtol=1e-5, atol=1e-5)


# Every logarithmic score is jittable. (name, args); args become jax arrays.
_ENS_OBS = [0.2, 1.1]
_ENS_FCT = [[0.1, -0.5, 0.9, 1.3], [0.8, 1.6, 0.2, 1.0]]
LOGS_CASES = [
    ("logs_ensemble", (_ENS_OBS, _ENS_FCT)),
    ("clogs_ensemble", (_ENS_OBS, _ENS_FCT, -1.0, 1.0)),
    ("logs_beta", ([0.3, 0.6], 0.7, 1.1, 0.0, 1.0)),
    ("logs_binomial", ([8.0, 3.0], 10.0, 0.6)),
    ("logs_exponential", ([0.3, 1.2], 0.9)),
    ("logs_exponential2", ([0.3, 1.2], 0.1, 2.0)),
    ("logs_2pexponential", ([0.3, -1.2], 0.5, 2.0, 0.1)),
    ("logs_gamma", ([0.2, 1.5], 1.1, 0.7)),
    ("logs_gev", ([0.3, 1.5], 0.3, 0.1, 1.2)),
    ("logs_gpd", ([0.3, 1.5], 0.3, 0.1, 1.2)),
    ("logs_hypergeometric", ([5.0, 6.0], 7.0, 13.0, 12.0)),
    ("logs_laplace", ([0.3, -1.5], 0.1, 1.2)),
    ("logs_loglaplace", ([0.3, 1.5], 0.1, 1.2)),
    ("logs_logistic", ([0.3, -1.5], 0.1, 1.2)),
    ("logs_loglogistic", ([0.3, 1.5], 0.1, 1.2)),
    ("logs_lognormal", ([0.3, 1.5], 0.1, 1.2)),
    ("logs_mixnorm", ([0.3, 1.5], [0.0, -2.9, 0.9], [0.5, 1.4, 0.7])),
    ("logs_negbinom", ([2.0, 4.0], 7.0, 0.8)),
    ("logs_normal", ([0.3, -1.5], 0.1, 1.2)),
    ("logs_2pnormal", ([0.3, -1.5], 0.5, 2.0, 0.1)),
    ("logs_poisson", ([1.0, 4.0], 3.0)),
    ("logs_t", ([0.3, -1.5], 4.0, 0.1, 1.2)),
    ("logs_tlogistic", ([0.3, 1.5], 0.1, 1.2, -1.0, 2.0)),
    ("logs_tnormal", ([0.3, 1.5], 0.1, 1.2, -1.0, 2.0)),
    ("logs_tt", ([0.3, 1.5], 4.0, 0.1, 1.2, -1.0, 2.0)),
    ("logs_uniform", ([0.3, 1.5], -1.0, 2.0)),
]


@pytest.mark.parametrize("name, args", LOGS_CASES, ids=[c[0] for c in LOGS_CASES])
def test_logs_jit(name, args):
    fn = getattr(sr, name)
    args = [jnp.asarray(a) for a in args]
    np.testing.assert_allclose(jax.jit(fn)(*args), fn(*args), rtol=1e-6)


@pytest.mark.parametrize("estimator", ["nrg", "fair", "akr", "akr_circperm"])
def test_energy_score_jit(estimator):
    rng = np.random.default_rng(0)  # float32 rounding: keep the draw fixed
    obs = jnp.asarray(rng.normal(size=(5, 3)))
    fct = jnp.asarray(rng.normal(size=(5, 11, 3)))

    @jax.jit
    def es(o, x):
        return sr.es_ensemble(o, x, estimator=estimator)

    @jax.jit
    def twes(o, x):
        return sr.twes_ensemble(o, x, lambda v: v.clip(min=-0.5), estimator=estimator)

    expected = sr.es_ensemble(obs, fct, estimator=estimator)
    np.testing.assert_allclose(es(obs, fct), expected, rtol=1e-5, atol=1e-5)
    expected = sr.twes_ensemble(
        obs, fct, lambda v: v.clip(min=-0.5), estimator=estimator
    )
    np.testing.assert_allclose(twes(obs, fct), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("estimator", ["nrg", "fair"])
def test_variogram_score_jit(estimator):
    rng = np.random.default_rng(0)  # float32 rounding: keep the draw fixed
    obs = jnp.asarray(rng.normal(size=(5, 3)))
    fct = jnp.asarray(rng.normal(size=(5, 11, 3)))

    @jax.jit
    def vs(o, x):
        return sr.vs_ensemble(o, x, estimator=estimator)

    @jax.jit
    def twvs(o, x):
        return sr.twvs_ensemble(o, x, lambda v: v.clip(min=-0.5), estimator=estimator)

    expected = sr.vs_ensemble(obs, fct, estimator=estimator)
    np.testing.assert_allclose(vs(obs, fct), expected, rtol=1e-5, atol=1e-5)
    expected = sr.twvs_ensemble(
        obs, fct, lambda v: v.clip(min=-0.5), estimator=estimator
    )
    np.testing.assert_allclose(twvs(obs, fct), expected, rtol=1e-5, atol=1e-5)
