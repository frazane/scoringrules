import math

import numpy as np
import pytest
import scoringrules as sr

from ._parity import (
    MV_CASES,
    UV_CASES,
    mv_ensemble_inputs,
    mv_v_func,
    mv_w_func,
    uv_ensemble_inputs,
    uv_w_func,
)
from .conftest import assert_inferred


ENSEMBLE_SIZE = 11
N = 10
N_VARS = 3

ESTIMATORS = ["nrg", "fair", "akr", "akr_circperm"]

UV_SCORES = [
    "gksuv_ensemble",
    "twgksuv_ensemble",
    "owgksuv_ensemble",
    "vrgksuv_ensemble",
]
MV_SCORES = [
    "gksmv_ensemble",
    "twgksmv_ensemble",
    "owgksmv_ensemble",
    "vrgksmv_ensemble",
]

# reference data for the weighted univariate scores
UV_FCT = np.array(
    [
        [-0.03574194, 0.06873582, 0.03098684, 0.07316138, 0.08498165],
        [-0.11957874, 0.26472238, -0.06324622, 0.43026451, -0.25640457],
        [-1.31340831, -1.43722153, -1.01696021, -0.70790148, -1.20432392],
        [1.26054027, 1.03477874, 0.61688454, 0.75305795, 1.19364529],
        [0.63192933, 0.33521695, 0.20626017, 0.43158264, 0.69518928],
        [0.83305233, 0.71965134, 0.96687378, 1.0000473, 0.82714425],
        [0.0128655, 0.03849841, 0.02459106, -0.02618909, 0.18687008],
        [-0.69399216, -0.59432627, -1.43629972, 0.01979857, -0.3286767],
        [1.92958764, 2.2678255, 1.86036922, 1.84766384, 2.03331138],
        [0.80847492, 1.11265193, 0.58995365, 1.04838184, 1.10439277],
    ]
)
UV_OBS = np.array(
    [
        0.19640722,
        -0.11300369,
        -1.0400268,
        0.84306533,
        0.36572078,
        0.82487264,
        0.14088773,
        -0.51936113,
        1.70706264,
        0.75985908,
    ]
)

# reference data for the weighted multivariate scores
MV_FCT = np.array(
    [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
).T
MV_OBS = np.array([0.2743836, 0.8146400])


def assert_close(result, expected, backend, *, rtol=1e-7, atol=1e-6):
    """Compare a framework-native result to a numpy/scalar value; jax runs in
    float32 in the test environment, so its tolerances are loosened."""
    if backend == "jax":
        rtol, atol = max(rtol, 1e-4), max(atol, 1e-5)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=rtol, atol=atol)


def _uv_inputs(rng=np.random):
    obs = rng.randn(N)
    mu = obs + rng.randn(N) * 0.1
    sigma = abs(rng.randn(N)) * 0.3
    fct = rng.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]
    return obs, fct


def _mv_inputs():
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    return obs, fct


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_gksuv(estimator, backend, to_backend, backend_kwargs):
    kwargs = {"estimator": estimator, **backend_kwargs}
    obs, fct = _uv_inputs()

    # undefined estimator
    with pytest.raises(ValueError):
        sr.gksuv_ensemble(
            to_backend(obs),
            to_backend(fct),
            estimator="undefined_estimator",
            **backend_kwargs,
        )

    # shape and m_axis keyword
    res = sr.gksuv_ensemble(to_backend(obs), to_backend(fct), **kwargs)
    assert_inferred(res, backend)
    assert res.shape == (N,)
    res2 = sr.gksuv_ensemble(to_backend(obs), to_backend(fct.T), m_axis=0, **kwargs)
    assert_inferred(res2, backend)
    assert_close(res2, np.asarray(res), backend)

    if estimator == "nrg":
        # non-negative values
        assert not np.any(np.asarray(res) < 0.0)
        res = sr.gksuv_ensemble(
            to_backend(obs),
            to_backend(np.random.randn(ENSEMBLE_SIZE, N)),
            m_axis=0,
            **kwargs,
        )
        assert not np.any(np.asarray(res) < 0.0)

        # approx zero when perfect forecast
        perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
        res = sr.gksuv_ensemble(to_backend(obs), to_backend(perfect_fct), **kwargs)
        assert not np.any(np.asarray(res) - 0.0 > 0.0001)

    # test correctness
    obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])
    expected = {
        "nrg": 0.2490516,
        "fair": 0.1737752,
        "akr": 0.2464915,
        "akr_circperm": 0.1010588,
    }
    res = sr.gksuv_ensemble(to_backend(obs), to_backend(fct), **kwargs)
    assert_close(res, expected[estimator], backend)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_gksmv(estimator, backend, to_backend, backend_kwargs):
    kwargs = {"estimator": estimator, **backend_kwargs}
    obs, fct = _mv_inputs()

    # undefined estimator
    with pytest.raises(ValueError):
        sr.gksmv_ensemble(
            to_backend(obs),
            to_backend(fct),
            estimator="undefined_estimator",
            **backend_kwargs,
        )

    # incorrect dimensions
    with pytest.raises(ValueError):
        sr.gksmv_ensemble(to_backend(obs), to_backend(fct[..., :-1]), **kwargs)

    res = sr.gksmv_ensemble(to_backend(obs), to_backend(fct), **kwargs)
    assert_inferred(res, backend)
    assert res.shape == (N,)

    # approx zero when perfect forecast
    if estimator == "nrg":
        perfect_fct = (
            np.expand_dims(obs, axis=-2)
            + np.random.randn(N, ENSEMBLE_SIZE, N_VARS) * 0.00001
        )
        res = sr.gksmv_ensemble(to_backend(obs), to_backend(perfect_fct), **kwargs)
        assert not np.any(np.asarray(res) - 0.0 > 0.0001)

    # test correctness
    obs = np.array([11.6, -23.1])
    fct = np.array(
        [[9.8, 8.7, 11.9, 12.1, 13.4], [-24.8, -18.5, -29.9, -18.3, -21.0]]
    ).transpose()
    expected = {
        "nrg": 0.5868737,
        "fair": 0.4870162,
        "akr": 0.4874259,
        "akr_circperm": 0.4866066,
    }
    res = sr.gksmv_ensemble(to_backend(obs), to_backend(fct), **kwargs)
    assert_close(res, expected[estimator], backend)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_twgksuv(estimator, backend, to_backend, backend_kwargs):
    kwargs = {"estimator": estimator, **backend_kwargs}
    obs, fct = _uv_inputs()
    obs, fct = to_backend(obs), to_backend(fct)

    res = sr.gksuv_ensemble(obs, fct, **kwargs)

    # no argument given
    resw = sr.twgksuv_ensemble(obs, fct, **kwargs)
    assert_inferred(resw, backend)
    assert_close(resw, np.asarray(res), backend, rtol=1e-10)

    # a and b
    resw = sr.twgksuv_ensemble(obs, fct, a=float("-inf"), b=float("inf"), **kwargs)
    assert_close(resw, np.asarray(res), backend, rtol=1e-10)

    # v_func as identity function
    resw = sr.twgksuv_ensemble(obs, fct, v_func=lambda x: x, **kwargs)
    assert_close(resw, np.asarray(res), backend, rtol=1e-10)

    # test correctness
    if estimator != "nrg":
        return
    obs, fct = to_backend(UV_OBS), to_backend(UV_FCT)
    cases = [
        ({"v_func": lambda x: x.clip(min=-1.0)}, 0.01018774),
        ({"a": -1.0}, 0.01018774),
        ({"v_func": lambda x: x.clip(max=1.85)}, 0.0089314),
        ({"b": 1.85}, 0.0089314),
    ]
    for tw_kwargs, expected in cases:
        res = sr.twgksuv_ensemble(obs, fct, **tw_kwargs, **kwargs)
        assert_inferred(res, backend)
        assert_close(np.mean(np.asarray(res)), expected, backend, rtol=1e-6)


def test_twgksmv(backend, to_backend, backend_kwargs):
    obs, fct = _mv_inputs()
    obs, fct = to_backend(obs), to_backend(fct)

    res0 = sr.gksmv_ensemble(obs, fct, **backend_kwargs)
    res = sr.twgksmv_ensemble(obs, fct, lambda x: x, **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend, rtol=1e-6)

    obs, fct = to_backend(MV_OBS), to_backend(MV_FCT)
    res = sr.twgksmv_ensemble(obs, fct, lambda x: x.clip(min=0.2), **backend_kwargs)
    assert_close(res, 0.08671498, backend, rtol=1e-6)

    # a threshold that moves obs and two members (every MV_OBS/MV_FCT value is < 1)
    expected = _ref_kernel_score_mv(MV_OBS, MV_FCT, v=lambda x: min(x, 0.7))
    res = sr.twgksmv_ensemble(obs, fct, lambda x: x.clip(max=0.7), **backend_kwargs)
    assert_close(res, expected, backend, rtol=1e-6)


@pytest.mark.parametrize("score", ["owgksuv_ensemble", "vrgksuv_ensemble"])
def test_weighted_gksuv(score, backend, to_backend, backend_kwargs):
    fn = getattr(sr, score)
    obs, fct = _uv_inputs()

    # m_axis keyword
    res = fn(
        to_backend(obs),
        to_backend(np.random.randn(ENSEMBLE_SIZE, N)),
        w_func=lambda x: x * 0.0 + 1.0,
        m_axis=0,
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert not np.any(np.asarray(res) < 0.0)

    obs, fct = to_backend(obs), to_backend(fct)
    res = np.asarray(sr.gksuv_ensemble(obs, fct, estimator="nrg", **backend_kwargs))

    # no argument given, a and b, w_func as a constant function
    for w_kwargs in (
        {},
        {"a": float("-inf"), "b": float("inf")},
        {"w_func": lambda x: x * 0.0 + 1.0},
    ):
        resw = fn(obs, fct, **w_kwargs, **backend_kwargs)
        assert_inferred(resw, backend)
        assert_close(resw, res, backend, rtol=1e-5, atol=1e-6)

    # test correctness
    expected = {
        "owgksuv_ensemble": (0.01036335, 0.008905213),
        "vrgksuv_ensemble": (0.01476682, 0.04011836),
    }[score]
    obs, fct = to_backend(UV_OBS), to_backend(UV_FCT)
    cases = [
        ({"w_func": lambda x: (x > -1) * 1.0}, expected[0]),
        ({"a": -1.0}, expected[0]),
        ({"w_func": lambda x: (x < 1.85) * 1.0}, expected[1]),
        ({"b": 1.85}, expected[1]),
    ]
    for w_kwargs, value in cases:
        res = fn(obs, fct, **w_kwargs, **backend_kwargs)
        assert_close(np.mean(np.asarray(res)), value, backend, rtol=1e-5)


@pytest.mark.parametrize("score", ["owgksmv_ensemble", "vrgksmv_ensemble"])
def test_weighted_gksmv(score, backend, to_backend, backend_kwargs):
    fn = getattr(sr, score)
    obs, fct = _mv_inputs()
    obs, fct = to_backend(obs), to_backend(fct)

    res0 = sr.gksmv_ensemble(obs, fct, **backend_kwargs)
    res = fn(obs, fct, lambda x: x.mean() * 0.0 + 1.0, **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend, rtol=1e-6)

    # weight functions that drop members (every MV_OBS/MV_FCT value is < 1)
    kind = "ow" if score == "owgksmv_ensemble" else "vr"
    obs, fct = to_backend(MV_OBS), to_backend(MV_FCT)
    for w_func, w_ref in (
        (lambda x: (x > 0.2).all(), lambda x: float(all(xi > 0.2 for xi in x))),
        (lambda x: x[1] > 0.5, lambda x: float(x[1] > 0.5)),
        (lambda x: 1.0 / (1.0 + x.sum() ** 2), lambda x: 1.0 / (1.0 + sum(x) ** 2)),
    ):
        expected = _ref_kernel_score_mv(MV_OBS, MV_FCT, w=w_ref, kind=kind)
        res = fn(obs, fct, w_func, **backend_kwargs)
        assert_close(res, expected, backend, rtol=1e-6)


def _ref_kernel_score_mv(obs, fct, v=None, w=None, kind="tw"):
    """Plain-Python Gaussian kernel score of one multivariate ensemble ``fct``
    (members x variables) from the definitions in the docstrings: ``"tw"``
    applies the chaining function ``v`` elementwise, ``"ow"``/``"vr"`` use
    the weight function ``w`` of a member vector."""
    obs, fct = [float(y) for y in obs], [[float(x) for x in m] for m in fct]
    M = len(fct)

    def k(a, b):
        return math.exp(-0.5 * sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    if kind == "tw":
        v = v or (lambda x: x)
        obs, fct = [v(y) for y in obs], [[v(x) for x in m] for m in fct]
        wy, wx = 1.0, [1.0] * M
    else:
        wy, wx = w(obs), [w(m) for m in fct]
    e_1 = sum(k(fct[m], obs) * wx[m] * wy for m in range(M)) / M
    e_2 = sum(k(fct[m], fct[j]) * wx[m] * wx[j] for m in range(M) for j in range(M)) / (
        M**2
    )
    e_3 = k(obs, obs) * wy * wy
    if kind == "ow":
        wbar = sum(wx) / M
        return -(e_1 / wbar - 0.5 * e_2 * wy / wbar**2 - 0.5 * k(obs, obs) * wy)
    return -(e_1 - 0.5 * e_2 - 0.5 * e_3)


def test_ref_kernel_score_mv_matches_r():
    """The plain-Python reference reproduces the R (scoringRules) values used
    as known values in this file."""
    np.testing.assert_allclose(
        _ref_kernel_score_mv(MV_OBS, MV_FCT, v=lambda x: max(x, 0.2)),
        0.08671498,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _ref_kernel_score_mv(
            MV_OBS, MV_FCT, w=lambda x: float(all(xi > 0.2 for xi in x)), kind="ow"
        ),
        0.03901075,
        rtol=1e-6,
    )


# the univariate kernel scores have no nan_policy
_KS_UV_CASES = [c for c in UV_CASES if not c.startswith("nan_")]


@pytest.mark.parametrize("case", _KS_UV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["gksuv_ensemble", "twgksuv_ensemble"])
def test_gksuv_paths_agree(score, estimator, weighted, M, case, paths_agree):
    obs, fct, kwargs = uv_ensemble_inputs(case, M, weighted)
    args = (-0.5, 1.0) if score == "twgksuv_ensemble" else ()
    paths_agree(getattr(sr, score), obs, fct, *args, estimator=estimator, **kwargs)


@pytest.mark.parametrize("case", _KS_UV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owgksuv_ensemble", "vrgksuv_ensemble"])
def test_weighted_gksuv_paths_agree(score, weighted, M, case, paths_agree):
    obs, fct, kwargs = uv_ensemble_inputs(case, M, weighted)
    paths_agree(getattr(sr, score), obs, fct, w_func=uv_w_func, **kwargs)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["gksmv_ensemble", "twgksmv_ensemble"])
def test_gksmv_paths_agree(score, estimator, weighted, M, case, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    args = (mv_v_func,) if score == "twgksmv_ensemble" else ()
    paths_agree(getattr(sr, score), obs, fct, *args, estimator=estimator, **kwargs)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owgksmv_ensemble", "vrgksmv_ensemble"])
def test_weighted_gksmv_paths_agree(score, weighted, M, case, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    paths_agree(getattr(sr, score), obs, fct, mv_w_func, **kwargs)


def _uv_w_func(x):
    return 1.0 / (1.0 + x**2)


def _mv_w_func(x):
    return 1.0 / (1.0 + x.mean() ** 2)


def _gks_ens_w_inputs(mv, seed=0):
    """Inputs plus integer member counts and the ensemble with duplicated members."""
    rng = np.random.default_rng(seed)
    M = 6
    obs = rng.normal(size=(N, N_VARS) if mv else (N,))
    fct = rng.normal(size=(N, M, N_VARS) if mv else (N, M))
    counts = np.array([1, 3, 1, 2, 1, 1])
    fct_dup = np.repeat(fct, counts, axis=1)
    ens_w = np.broadcast_to(counts, (N, M)).astype(float)
    return obs, fct, counts, fct_dup, ens_w


_GKS_W_SCORES = [
    *[("gksuv_ensemble", est) for est in ESTIMATORS],
    *[("gksmv_ensemble", est) for est in ESTIMATORS],
    ("owgksuv_ensemble", None),
    ("vrgksuv_ensemble", None),
    ("owgksmv_ensemble", None),
    ("vrgksmv_ensemble", None),
]


def _gks_call(fn, score, estimator, obs, fct, **kwargs):
    if estimator is not None:
        return np.asarray(fn(obs, fct, estimator=estimator, **kwargs))
    if score.endswith("uv_ensemble"):
        return np.asarray(fn(obs, fct, w_func=_uv_w_func, **kwargs))
    return np.asarray(fn(obs, fct, _mv_w_func, **kwargs))


@pytest.mark.parametrize("score,estimator", _GKS_W_SCORES)
def test_gks_uniform_ens_w(score, estimator, native, backend_kwargs):
    """Uniform ens_w reproduces the unweighted score."""
    fn = native(getattr(sr, score))
    obs, fct, _, _, ens_w = _gks_ens_w_inputs(score.endswith("mv_ensemble"))
    ens_w = np.ones_like(ens_w)
    res = _gks_call(fn, score, estimator, obs, fct, **backend_kwargs)
    res_w = _gks_call(fn, score, estimator, obs, fct, ens_w=ens_w, **backend_kwargs)
    np.testing.assert_allclose(res_w, res, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "score,estimator", [s for s in _GKS_W_SCORES if s[1] in (None, "nrg")]
)
def test_gks_ens_w_duplicated_members(score, estimator, native, backend_kwargs):
    """Integer ens_w equals the unweighted score of the ensemble with each member
    repeated that many times (holds for the energy form, not for fair or akr)."""
    fn = native(getattr(sr, score))
    obs, fct, _, fct_dup, ens_w = _gks_ens_w_inputs(score.endswith("mv_ensemble"))
    res_w = _gks_call(fn, score, estimator, obs, fct, ens_w=ens_w, **backend_kwargs)
    res_dup = _gks_call(fn, score, estimator, obs, fct_dup, **backend_kwargs)
    np.testing.assert_allclose(res_w, res_dup, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("score", ["gksuv_ensemble", "gksmv_ensemble"])
def test_gks_fair_ens_w_known_value(score, native, backend_kwargs):
    """Weighted fair estimator excludes the i == j pairs (#150)."""
    mv = score == "gksmv_ensemble"
    obs = np.array([0.3, -0.1]) if mv else np.array(0.3)
    fct = (
        np.array([[0.1, 0.2], [-0.4, 0.0], [1.2, -0.5]])
        if mv
        else np.array([0.1, -0.4, 1.2])
    )
    w = np.array([0.5, 0.3, 0.2])
    kern = lambda a, b: np.exp(-0.5 * np.sum((np.asarray(a) - b) ** 2))  # noqa: E731
    e_1 = sum(w[i] * kern(fct[i], obs) for i in range(3))
    e_2 = sum(
        w[i] * w[j] * kern(fct[i], fct[j]) for i in range(3) for j in range(3) if i != j
    )
    expected = -(e_1 - 0.5 * e_2 / (1 - np.sum(w**2)) - 0.5)
    res = native(getattr(sr, score))(
        obs, fct, ens_w=w, estimator="fair", **backend_kwargs
    )
    # rtol for jax, which runs in float32 here
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)


@pytest.mark.parametrize("score", ["gksuv_ensemble", "gksmv_ensemble"])
def test_gks_akr_circperm_ens_w_known_value(score, native, backend_kwargs):
    """Weighted akr_circperm pairs member i with (i + M // 2) % M and uses w_i (#149)."""
    mv = score == "gksmv_ensemble"
    obs = [0.3, -0.1] if mv else [0.3]
    fct = [[0.1, 0.2], [-0.4, 0.0], [1.2, -0.5]] if mv else [[0.1], [-0.4], [1.2]]
    w = [0.5, 0.3, 0.2]
    partner = [1, 2, 0]  # M = 3, M // 2 = 1

    def kern(a, b):
        return math.exp(-0.5 * sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    e_1 = sum(w[i] * kern(fct[i], obs) for i in range(3))
    e_2 = sum(w[i] * kern(fct[i], fct[partner[i]]) for i in range(3))
    expected = -(e_1 - 0.5 * e_2 - 0.5)
    obs_in = np.array(obs) if mv else np.array(obs[0])
    fct_in = np.array(fct) if mv else np.array(fct)[:, 0]
    res = native(getattr(sr, score))(
        obs_in, fct_in, ens_w=np.array(w), estimator="akr_circperm", **backend_kwargs
    )
    # rtol for jax, which runs in float32 here
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)


def _mv_score_args(score, estimator):
    """Positional extra args and kwargs for a multivariate kernel score."""
    if score == "gksmv_ensemble":
        return (), {"estimator": estimator}
    if score == "twgksmv_ensemble":
        return (lambda x: x,), {"estimator": estimator}
    return (_mv_w_func,), {}


_MV_LAYOUT_CASES = [
    *[(s, est) for s in ("gksmv_ensemble", "twgksmv_ensemble") for est in ESTIMATORS],
    ("owgksmv_ensemble", None),
    ("vrgksmv_ensemble", None),
]


@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score,estimator", _MV_LAYOUT_CASES)
def test_gksmv_ens_w_v_axis_first(score, estimator, weighted, native, backend_kwargs):
    """ens_w follows fct without its variables axis, also when v_axis comes
    before m_axis (it used to raise: the weight axis was taken against fct.ndim)."""
    obs, fct, kwargs = mv_ensemble_inputs("default", 7, weighted)
    args, extra = _mv_score_args(score, estimator)
    fn = native(getattr(sr, score))
    expected = fn(obs, fct, *args, **extra, **kwargs, **backend_kwargs)
    kwargs = {**kwargs, "m_axis": -1, "v_axis": -2}
    fct_t = np.swapaxes(fct, -1, -2).copy()
    res = fn(obs, fct_t, *args, **extra, **kwargs, **backend_kwargs)
    np.testing.assert_allclose(np.asarray(res), np.asarray(expected), rtol=1e-6)
    # m_axis=0 with v_axis=1: fct (M, D, N), obs (D, N), ens_w (M, N)
    fct_t = np.moveaxis(fct, (-2, -1), (0, 1)).copy()
    kwargs = {**kwargs, "m_axis": 0, "v_axis": 1}
    if weighted:
        kwargs["ens_w"] = np.moveaxis(kwargs["ens_w"], -1, 0).copy()
    res = fn(obs.T.copy(), fct_t, *args, **extra, **kwargs, **backend_kwargs)
    np.testing.assert_allclose(np.asarray(res), np.asarray(expected), rtol=1e-6)


@pytest.mark.parametrize("score,estimator", _MV_LAYOUT_CASES)
def test_gksmv_ens_w_silent_layout(score, estimator, native, backend_kwargs):
    """fct (D, M, N) with M == N, m_axis=1, v_axis=0: ens_w (M, N) used to be
    moved along the wrong axis without an error, so the score used the wrong
    weights. Compare with the default layout on moveaxis'd inputs."""
    rng = np.random.default_rng(5)
    D, M = 3, 5
    obs = rng.normal(size=(M, D))  # (N, D) with N == M
    fct = rng.normal(size=(M, M, D))  # (N, M, D)
    ens_w = rng.uniform(0.1, 2.0, size=(M, M))  # (N, M)
    args, extra = _mv_score_args(score, estimator)
    fn = native(getattr(sr, score))
    expected = fn(obs, fct, *args, ens_w=ens_w, **extra, **backend_kwargs)
    res = fn(
        obs.T.copy(),
        np.moveaxis(fct, (0, 1, 2), (2, 1, 0)).copy(),
        *args,
        m_axis=1,
        v_axis=0,
        ens_w=ens_w.T.copy(),
        **extra,
        **backend_kwargs,
    )
    np.testing.assert_allclose(np.asarray(res), np.asarray(expected), rtol=1e-6)


def test_kernels_legacy_backend_string_deprecated():
    obs, fct = np.random.randn(N), np.random.randn(N, ENSEMBLE_SIZE)
    obs_mv = np.random.randn(N, N_VARS)
    fct_mv = np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    calls = [
        lambda: sr.gksuv_ensemble(obs, fct, backend="numpy"),
        lambda: sr.twgksuv_ensemble(obs, fct, backend="numpy"),
        lambda: sr.owgksuv_ensemble(obs, fct, backend="numpy"),
        lambda: sr.vrgksuv_ensemble(obs, fct, backend="numpy"),
        lambda: sr.gksmv_ensemble(obs_mv, fct_mv, backend="numpy"),
        lambda: sr.twgksmv_ensemble(obs_mv, fct_mv, lambda x: x, backend="numpy"),
        lambda: sr.owgksmv_ensemble(obs_mv, fct_mv, _mv_w_func, backend="numpy"),
        lambda: sr.vrgksmv_ensemble(obs_mv, fct_mv, _mv_w_func, backend="numpy"),
    ]
    for call in calls:
        with pytest.warns(DeprecationWarning) as record:
            call()
        # one warning, pointing at the caller (this file), not at the wrapper
        assert len(record) == 1 and record[0].filename == __file__


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_kernels_float32(estimator, backend, to_backend_float32, backend_kwargs):
    rng = np.random.default_rng(3)
    obs = to_backend_float32(rng.normal(size=N))
    fct = to_backend_float32(rng.normal(size=(N, ENSEMBLE_SIZE)))
    obs_mv = to_backend_float32(rng.normal(size=(N, N_VARS)))
    fct_mv = to_backend_float32(rng.normal(size=(N, ENSEMBLE_SIZE, N_VARS)))
    ens_w = to_backend_float32(rng.uniform(0.1, 1.0, (N, ENSEMBLE_SIZE)))
    kwargs = {"estimator": estimator, **backend_kwargs}
    for res in (
        sr.gksuv_ensemble(obs, fct, **kwargs),
        sr.gksuv_ensemble(obs, fct, ens_w=ens_w, **kwargs),
        sr.twgksuv_ensemble(obs, fct, v_func=lambda x: x.clip(min=-0.5), **kwargs),
        sr.owgksuv_ensemble(obs, fct, w_func=_uv_w_func, **backend_kwargs),
        sr.owgksuv_ensemble(obs, fct, b=0.5, ens_w=ens_w, **backend_kwargs),
        sr.owgksuv_ensemble(obs, fct, w_func=lambda x: x > 0.0, **backend_kwargs),
        sr.owgksuv_ensemble(
            obs, fct, w_func=lambda x: (x > 0.0) * 1.0, ens_w=ens_w, **backend_kwargs
        ),
        sr.vrgksuv_ensemble(obs, fct, w_func=lambda x: x > 0.0, **backend_kwargs),
    ):
        assert_inferred(res, backend)
        assert res.dtype == fct.dtype
    for res in (
        sr.gksmv_ensemble(obs_mv, fct_mv, **kwargs),
        sr.gksmv_ensemble(obs_mv, fct_mv, ens_w=ens_w, **kwargs),
        sr.twgksmv_ensemble(obs_mv, fct_mv, lambda x: x.clip(min=-0.5), **kwargs),
        sr.owgksmv_ensemble(obs_mv, fct_mv, _mv_w_func, **backend_kwargs),
        sr.vrgksmv_ensemble(
            obs_mv, fct_mv, lambda x: (x > 0.0).all(), ens_w=ens_w, **backend_kwargs
        ),
    ):
        assert_inferred(res, backend)
        assert res.dtype == fct_mv.dtype


def _finite_difference(f, x, index, h=1e-6):
    """Central difference of the scalar ``f`` at ``x`` along one entry."""
    x_plus, x_minus = x.clone(), x.clone()
    x_plus[index] += h
    x_minus[index] -= h
    return (f(x_plus) - f(x_minus)).item() / (2 * h)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("mv", [False, True], ids=["uv", "mv"])
def test_kernels_grad_torch(estimator, mv):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    shape = (4, ENSEMBLE_SIZE, N_VARS) if mv else (4, ENSEMBLE_SIZE)
    obs = torch.as_tensor(rng.normal(size=shape[:1] + shape[2:]))
    fct = torch.tensor(rng.normal(size=shape), requires_grad=True)
    ens_w = torch.as_tensor(rng.uniform(0.1, 1.0, (4, ENSEMBLE_SIZE)))
    score = sr.gksmv_ensemble if mv else sr.gksuv_ensemble
    index = (1, 2, 0) if mv else (1, 2)
    for kwargs in ({}, {"ens_w": ens_w}):

        def f(x):
            return score(obs, x, estimator=estimator, **kwargs).sum()

        fct.grad = None
        f(fct).backward()
        assert fct.grad is not None and torch.isfinite(fct.grad).all()
        with torch.no_grad():
            expected = _finite_difference(f, fct.detach(), index)
        np.testing.assert_allclose(fct.grad[index].item(), expected, rtol=1e-5)


@pytest.mark.parametrize(
    "score",
    ["owgksuv_ensemble", "vrgksuv_ensemble", "owgksmv_ensemble", "vrgksmv_ensemble"],
)
def test_weighted_kernels_grad_torch(score):
    """Autograd w.r.t. fct and obs matches a central finite difference."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(6)
    mv = score.endswith("mv_ensemble")
    shape = (4, ENSEMBLE_SIZE, N_VARS) if mv else (4, ENSEMBLE_SIZE)
    obs = torch.tensor(rng.normal(size=shape[:1] + shape[2:]), requires_grad=True)
    fct = torch.tensor(rng.normal(size=shape), requires_grad=True)
    fn = getattr(sr, score)

    def f(o, x):
        if mv:
            return fn(o, x, _mv_w_func).sum()
        return fn(o, x, w_func=_uv_w_func).sum()

    f(obs, fct).backward()
    assert torch.isfinite(fct.grad).all() and torch.isfinite(obs.grad).all()
    fct_index = (2, 3, 1) if mv else (2, 3)
    obs_index = (1, 0) if mv else (1,)
    with torch.no_grad():
        o, x = obs.detach(), fct.detach()
        expected_fct = _finite_difference(lambda x_: f(o, x_), x, fct_index)
        expected_obs = _finite_difference(lambda o_: f(o_, x), o, obs_index)
    np.testing.assert_allclose(fct.grad[fct_index].item(), expected_fct, rtol=1e-5)
    np.testing.assert_allclose(obs.grad[obs_index].item(), expected_obs, rtol=1e-5)


def _jax_mv_scores(estimator):
    ens_w = np.random.default_rng(8).uniform(0.1, 1.0, (4, ENSEMBLE_SIZE))
    return {
        "gksmv": lambda o, x: sr.gksmv_ensemble(o, x, estimator=estimator),
        "gksmv_w": lambda o, x: sr.gksmv_ensemble(
            o, x, ens_w=ens_w, estimator=estimator
        ),
        "twgksmv": lambda o, x: sr.twgksmv_ensemble(
            o, x, lambda v: v.clip(min=-0.5), estimator=estimator
        ),
        "owgksmv": lambda o, x: sr.owgksmv_ensemble(o, x, _mv_w_func),
        "vrgksmv": lambda o, x: sr.vrgksmv_ensemble(o, x, _mv_w_func),
    }


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_kernels_mv_grad_jax(estimator):
    """jax grads of the multivariate scores are finite (the squared distance
    is a sum of squares; norm ** 2 had a NaN derivative at 0) and match a
    central finite difference, for fct and obs."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    rng = np.random.default_rng(7)
    with jax.enable_x64(True):
        obs = jnp.asarray(rng.normal(size=(4, N_VARS)))
        fct = jnp.asarray(rng.normal(size=(4, ENSEMBLE_SIZE, N_VARS)))
        h = 1e-6
        for name, score in _jax_mv_scores(estimator).items():

            def f(o, x):
                return score(o, x).sum()

            g_obs, g_fct = jax.grad(f, argnums=(0, 1))(obs, fct)
            assert jnp.isfinite(g_obs).all() and jnp.isfinite(g_fct).all(), name
            fd_fct = (
                f(obs, fct.at[1, 2, 0].add(h)) - f(obs, fct.at[1, 2, 0].add(-h))
            ) / (2 * h)
            fd_obs = (f(obs.at[1, 1].add(h), fct) - f(obs.at[1, 1].add(-h), fct)) / (
                2 * h
            )
            np.testing.assert_allclose(g_fct[1, 2, 0], fd_fct, rtol=1e-5, err_msg=name)
            np.testing.assert_allclose(g_obs[1, 1], fd_obs, rtol=1e-5, err_msg=name)
