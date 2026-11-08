import numpy as np
import pytest

import scoringrules as sr
from .conftest import assert_inferred

FCT_K = [
    [0.01, 0.32, 0.44, 0.23],
    [0.12, 0.05, 0.48, 0.35],
    [0.09, 0.21, 0.05, 0.65],
    [0.57, 0.31, 0.08, 0.04],
]
OBS_K = [1, 3, 4, 2]


def test_brier(backend, to_backend):
    # test exceptions
    with pytest.raises(ValueError):
        sr.brier_score(to_backend(1), to_backend(1.1))

    with pytest.raises(ValueError):
        sr.brier_score(to_backend(1), to_backend(-0.1))

    with pytest.raises(ValueError):
        sr.brier_score(to_backend(0.5), to_backend(0.5))

    with pytest.raises(ValueError):
        sr.brier_score(to_backend([0, np.nan]), to_backend([0.5, 0.5]))

    # test correctness
    res = sr.brier_score(to_backend(0), to_backend(0.271))
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 0.073441)

    res = sr.brier_score(to_backend([0, 1]), to_backend([0.271, 0.271]))
    assert_inferred(res, backend)
    assert np.allclose(np.asarray(res), [0.073441, 0.531441])


def test_rps(backend, to_backend):
    # test exceptions
    with pytest.raises(ValueError):
        sr.rps_score(to_backend(1), to_backend(1.1))

    with pytest.raises(ValueError):
        sr.rps_score(to_backend(1), to_backend(-0.1))

    with pytest.raises(ValueError):
        sr.rps_score(to_backend([1, 2]), to_backend(FCT_K))

    # test correctness
    obs, fct = 3, [0.01, 0.32, 0.44, 0.23]
    res = sr.rps_score(to_backend(obs), to_backend(fct))
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 0.1619)

    # onehot gives the same result
    res_onehot = sr.rps_score(to_backend([0, 0, 1, 0]), to_backend(fct), onehot=True)
    assert_inferred(res_onehot, backend)
    assert np.isclose(np.asarray(res_onehot), 0.1619)

    # k_axis
    res1 = sr.rps_score(to_backend(OBS_K), to_backend(FCT_K))
    res2 = sr.rps_score(to_backend(OBS_K), to_backend(np.transpose(FCT_K)), k_axis=0)
    assert_inferred(res1, backend)
    assert_inferred(res2, backend)
    assert res1.shape == (4,)
    assert np.allclose(np.asarray(res1), np.asarray(res2))

    onehot = np.eye(4)[np.asarray(OBS_K) - 1]
    res3 = sr.rps_score(
        to_backend(onehot.T),
        to_backend(np.transpose(FCT_K)),
        k_axis=0,
        onehot=True,
    )
    assert np.allclose(np.asarray(res1), np.asarray(res3))


def test_logs(backend, to_backend):
    # test exceptions
    with pytest.raises(ValueError):
        sr.log_score(to_backend(1), to_backend(1.1))

    with pytest.raises(ValueError):
        sr.log_score(to_backend(1), to_backend(-0.1))

    with pytest.raises(ValueError):
        sr.log_score(to_backend(0.5), to_backend(0.5))

    # test correctness
    res = sr.log_score(to_backend(0), to_backend(0.481))
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 0.6558514)

    res = sr.log_score(to_backend([0, 1]), to_backend([0.481, 0.481]))
    assert_inferred(res, backend)
    assert np.allclose(np.asarray(res), [0.6558514, -np.log(0.481)])


def test_rls(backend, to_backend):
    # test exceptions
    with pytest.raises(ValueError):
        sr.rls_score(to_backend(1), to_backend(1.1))

    with pytest.raises(ValueError):
        sr.rls_score(to_backend(1), to_backend(-0.1))

    # test correctness
    obs, fct = 2, [0.21, 0.31, 0.32, 0.16]
    res = sr.rls_score(to_backend(obs), to_backend(fct))
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 1.064002)

    # k_axis
    res1 = sr.rls_score(to_backend(OBS_K), to_backend(FCT_K))
    res2 = sr.rls_score(to_backend(OBS_K), to_backend(np.transpose(FCT_K)), k_axis=0)
    assert_inferred(res1, backend)
    assert res1.shape == (4,)
    assert np.allclose(np.asarray(res1), np.asarray(res2))


def test_brier_family_float32(backend, to_backend_float32):
    obs_b = to_backend_float32([0, 1, 1])
    fct_b = to_backend_float32([0.2, 0.7, 0.4])
    obs_k = to_backend_float32(OBS_K)
    fct_k = to_backend_float32(FCT_K)
    for res in (
        sr.brier_score(obs_b, fct_b),
        sr.log_score(obs_b, fct_b),
        sr.rps_score(obs_k, fct_k),
        sr.rls_score(obs_k, fct_k),
        # Python obs take the forecast's dtype
        sr.brier_score(1, fct_b),
        sr.log_score([0, 1, 1], fct_b),
        sr.rps_score(OBS_K, fct_k),
        sr.rls_score(OBS_K, fct_k),
    ):
        assert_inferred(res, backend)
        assert res.dtype == fct_b.dtype


def test_brier_legacy_backend_string_deprecated():
    with pytest.warns(DeprecationWarning):
        sr.brier_score(np.array([0, 1]), np.array([0.2, 0.7]), backend="numpy")


def test_brier_family_grad_torch():
    torch = pytest.importorskip("torch")
    obs_b = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    obs_k = torch.tensor(OBS_K, dtype=torch.float64)

    # analytic: d/dF (F - y)^2 = 2 (F - y), d/dF -log|F + y - 1| = -1 / (F + y - 1)
    fct = torch.tensor([0.2, 0.7, 0.4], dtype=torch.float64, requires_grad=True)
    sr.brier_score(obs_b, fct).sum().backward()
    torch.testing.assert_close(fct.grad, 2 * (fct.detach() - obs_b))
    fct.grad = None
    sr.log_score(obs_b, fct).sum().backward()
    torch.testing.assert_close(fct.grad, -1 / (fct.detach() + obs_b - 1))

    # finite differences
    for fn, obs, fct in (
        (sr.brier_score, obs_b, [0.2, 0.7, 0.4]),
        (sr.log_score, obs_b, [0.2, 0.7, 0.4]),
        (sr.rps_score, obs_k, FCT_K),
        (sr.rls_score, obs_k, FCT_K),
    ):
        fct = torch.tensor(fct, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda f, fn=fn, o=obs: fn(o, f), (fct,))


@pytest.mark.parametrize("fn", ["brier_score", "log_score", "rps_score", "rls_score"])
def test_brier_family_numba_arg_infers(fn, backend, to_backend):
    # no gufunc: the deprecated backend="numba" means "infer", also for jax/torch
    obs, fct = (
        ([0, 1], [0.2, 0.7])
        if fn in ("brier_score", "log_score")
        else (
            OBS_K,
            FCT_K,
        )
    )
    with pytest.warns(DeprecationWarning, match="backend='numba'"):
        res = getattr(sr, fn)(to_backend(obs), to_backend(fct), backend="numba")
    assert_inferred(res, backend)
