import numpy as np
import pytest
import scipy.stats as st
import scoringrules as sr
from .conftest import assert_inferred


N = 100


def test_interval_score(backend, to_backend, backend_kwargs):
    fa = to_backend

    # basic functionality
    res = sr.interval_score(fa(0.1), fa(0.0), fa(0.4), fa(0.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert res.shape == ()
    assert np.isclose(np.asarray(res), 0.4)

    # shapes
    res = sr.interval_score(
        obs=fa([0.1, 0.2, 0.3]),
        lower=fa([0.0, 0.1, 0.2]),
        upper=fa([0.4, 0.3, 0.5]),
        alpha=fa(0.5),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert res.shape == (3,)
    assert np.allclose(np.asarray(res), [0.4, 0.2, 0.3])

    res = sr.interval_score(
        obs=fa(np.random.uniform(size=(10,))),
        lower=fa(np.ones((10, 5)) * 0.2),
        upper=fa(np.ones((10, 5)) * 0.8),
        alpha=fa(np.linspace(0.1, 0.9, 5)),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert res.shape == (10, 5)

    # one alpha level in a (K,) array drops the K axis; N=1 keeps the batch axis
    res = sr.interval_score(
        obs=fa([0.1, 0.9]),
        lower=fa([[0.2], [0.2]]),
        upper=fa([[0.8], [0.8]]),
        alpha=fa([0.5]),
        **backend_kwargs,
    )
    assert res.shape == (2,)
    assert np.allclose(np.asarray(res), [1.0, 1.0])
    res = sr.interval_score(fa([0.1]), fa([0.2]), fa([0.8]), fa(0.5), **backend_kwargs)
    assert res.shape == (1,)

    # raise ValueError
    with pytest.raises(ValueError):
        _ = sr.interval_score(
            obs=fa(np.random.uniform(size=(10,))),
            lower=fa(np.ones((10, 5)) * 0.2),
            upper=fa(np.ones((10, 4)) * 0.8),
            alpha=fa(np.linspace(0.1, 0.9, 5)),
            **backend_kwargs,
        )

    with pytest.raises(ValueError):
        _ = sr.interval_score(
            obs=fa(np.random.uniform(size=(10,))),
            lower=fa(np.ones((10, 5)) * 0.2),
            upper=fa(np.ones((10, 5)) * 0.8),
            alpha=fa(np.linspace(0.1, 0.9, 4)),
            **backend_kwargs,
        )


## We use Bracher et al (2021) Eq. (3) to test the WIS
def test_weighted_interval_score(backend, to_backend, backend_kwargs):
    obs = np.zeros(N)
    alpha = np.linspace(0.01, 0.99, 99)
    upper = st.norm(0, 1).ppf(np.tile(1 - alpha / 2, (N, 1)))
    lower = st.norm(0, 1).ppf(np.tile(alpha / 2, (N, 1)))
    obs, alpha, upper, lower = map(to_backend, (obs, alpha, upper, lower))

    WIS = sr.weighted_interval_score(obs, obs, lower, upper, alpha, **backend_kwargs)
    assert_inferred(WIS, backend)
    assert WIS.shape == (N,)
    CRPS = sr.crps_normal(obs, 0, 1)
    WIS, CRPS = map(np.asarray, (WIS, CRPS))

    assert np.all(1 - WIS / CRPS <= 0.001 * CRPS)


def test_weighted_interval_score_list_alpha(backend, to_backend, backend_kwargs):
    alpha = [0.2, 0.5, 0.8]
    obs = to_backend(np.zeros(3))
    lower, upper = to_backend(-np.ones((3, 3))), to_backend(np.ones((3, 3)))
    res = sr.weighted_interval_score(obs, obs, lower, upper, alpha, **backend_kwargs)
    assert_inferred(res, backend)
    expected = sr.weighted_interval_score(
        obs, obs, lower, upper, to_backend(alpha), **backend_kwargs
    )
    assert np.allclose(np.asarray(res), np.asarray(expected))


def test_interval_float32(backend, to_backend_float32, backend_kwargs):
    fa = to_backend_float32
    obs = fa(np.random.randn(5))
    alpha = fa(np.linspace(0.1, 0.9, 5))
    lower = fa(np.tile(-np.linspace(2.0, 0.1, 5), (5, 1)))
    upper = fa(np.tile(np.linspace(2.0, 0.1, 5), (5, 1)))
    for res in (
        sr.interval_score(obs, lower, upper, alpha, **backend_kwargs),
        sr.weighted_interval_score(obs, obs, lower, upper, alpha, **backend_kwargs),
        # Python-float parameters take the inputs' dtype
        sr.interval_score(obs, lower[:, 0], upper[:, 0], 0.1, **backend_kwargs),
        sr.weighted_interval_score(
            obs, obs, lower, upper, alpha, 0.3, **backend_kwargs
        ),
        sr.weighted_interval_score(
            obs, obs, lower, upper, list(np.linspace(0.1, 0.9, 5)), **backend_kwargs
        ),
    ):
        assert_inferred(res, backend)
        assert res.dtype == obs.dtype


def test_interval_legacy_backend_string_deprecated():
    with pytest.warns(DeprecationWarning):
        sr.interval_score(0.1, 0.0, 0.4, 0.5, backend="numpy")


def test_weighted_interval_score_grad_torch():
    torch = pytest.importorskip("torch")
    dt = torch.float64
    obs = torch.tensor([0.1, 1.5], dtype=dt)
    alpha = torch.tensor([0.2, 0.5], dtype=dt)
    lower = torch.tensor([[-1.0, -0.5], [-1.0, -0.5]], dtype=dt, requires_grad=True)
    upper = torch.tensor([[1.0, 0.5], [1.0, 0.5]], dtype=dt, requires_grad=True)
    # analytic: dIS/du = 1 - (2/a) 1{y > u}, dIS/dl = -1 + (2/a) 1{y < l}
    sr.interval_score(obs, lower, upper, alpha).sum().backward()
    y, a = obs[:, None], alpha
    expected_u = 1 - (2 / a) * (y > upper.detach())
    expected_l = -1 + (2 / a) * (y < lower.detach())
    torch.testing.assert_close(upper.grad, expected_u)
    torch.testing.assert_close(lower.grad, expected_l)
    # WIS with the default weights alpha/2, divided by K + 1/2
    lower.grad = upper.grad = None
    sr.weighted_interval_score(obs, obs, lower, upper, alpha).sum().backward()
    torch.testing.assert_close(upper.grad, expected_u * (a / 2) / 2.5)
    torch.testing.assert_close(lower.grad, expected_l * (a / 2) / 2.5)
    # finite differences, also through the median term
    median = torch.tensor([0.3, 1.0], dtype=dt, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda m, lo, up: sr.weighted_interval_score(obs, m, lo, up, alpha),
        (median, lower, upper),
    )


def test_interval_score_numba_arg_infers(backend, to_backend):
    # no gufunc: backend="numba" means "infer", also for jax/torch inputs
    fa = to_backend
    res = sr.interval_score(fa(0.1), fa(0.0), fa(0.4), fa(0.5), backend="numba")
    assert_inferred(res, backend)


def test_weighted_interval_score_numba_dispatch(backend, to_backend):
    args = [to_backend(x) for x in _wis_inputs("default")[0]]
    if backend in ("jax", "torch"):
        with pytest.raises(ValueError, match="numpy-compatible"):
            sr.weighted_interval_score(*args, backend="numba")
    else:
        pytest.importorskip("numba")
        assert_inferred(sr.weighted_interval_score(*args, backend="numba"), backend)


def test_weighted_interval_score_set_active_numba(requires_numba, monkeypatch):
    from scoringrules.backend import backends
    from scoringrules.core import interval

    calls = []
    gufunc = interval._weighted_interval_score_gufunc

    def spy(*args):
        calls.append(args)
        return gufunc(*args)

    monkeypatch.setattr(interval, "_weighted_interval_score_gufunc", spy)
    monkeypatch.setattr(backends, "_active", "numba")
    args, _ = _wis_inputs("default")
    sr.weighted_interval_score(*args)
    assert len(calls) == 1
    torch = pytest.importorskip("torch")
    res = sr.weighted_interval_score(*map(torch.as_tensor, args))
    assert len(calls) == 1 and isinstance(res, torch.Tensor)


# The array path adds ``w_median * median`` where the gufunc adds
# ``w_median * |obs - median|`` (#141), so the cases below set ``w_median=0`` to
# compare everything else, and ``median_term`` checks the median term alone.
WIS_CASES = (
    "default",
    "n1",
    "int_obs",
    "int_bounds",
    "scalar_obs",
    "batch",
    "list_inputs",
    "weights",
    "median_term",
)


def _wis_inputs(case, K=4, N=5, seed=0):
    rng = np.random.default_rng(seed)
    batch = {"n1": (1,), "batch": (2, N), "scalar_obs": ()}.get(case, (N,))
    obs = rng.normal(size=batch)
    if case == "int_obs":
        obs = rng.integers(-2, 3, size=batch)
    alpha = np.linspace(0.1, 0.9, K)
    half = rng.uniform(0.1, 2.0, size=(*batch, K))
    center = rng.normal(size=(*batch, 1))
    lower, upper = center - half, center + half
    if case == "int_bounds":
        lower, upper = np.floor(lower).astype(int), np.ceil(upper).astype(int)
    median = obs + rng.normal(size=batch)
    kwargs = {"w_median": 0.0}
    if case == "weights":
        kwargs["w_alpha"] = rng.uniform(0.1, 1.0, size=K)
    elif case == "median_term":
        kwargs = {}
    if case == "scalar_obs":
        obs, median = float(obs), float(median)
    args = (obs, median, lower, upper, alpha)
    if case == "list_inputs":
        args = tuple(np.asarray(a).tolist() for a in args)
    return args, kwargs


def _wis_parity_params():
    for case in WIS_CASES:
        marks = ()
        if case == "median_term":
            marks = pytest.mark.xfail(
                raises=AssertionError,
                strict=True,
                reason="#141: array WIS uses w_median * median, gufunc |obs - median|",
            )
        yield pytest.param(case, id=case, marks=marks)


@pytest.mark.parametrize("case", _wis_parity_params())
def test_weighted_interval_score_paths_agree(case, paths_agree):
    args, kwargs = _wis_inputs(case)
    paths_agree(sr.weighted_interval_score, *args, **kwargs)
