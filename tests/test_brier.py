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


def test_brier(backend, to_backend, backend_kwargs):
    # test exceptions
    with pytest.raises(ValueError):
        sr.brier_score(to_backend(1), to_backend(1.1), **backend_kwargs)

    with pytest.raises(ValueError):
        sr.brier_score(to_backend(1), to_backend(-0.1), **backend_kwargs)

    with pytest.raises(ValueError):
        sr.brier_score(to_backend(0.5), to_backend(0.5), **backend_kwargs)

    with pytest.raises(ValueError):
        sr.brier_score(to_backend([0, np.nan]), to_backend([0.5, 0.5]))

    # test correctness
    res = sr.brier_score(to_backend(0), to_backend(0.271), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 0.073441)

    res = sr.brier_score(to_backend([0, 1]), to_backend([0.271, 0.271]))
    assert_inferred(res, backend)
    assert np.allclose(np.asarray(res), [0.073441, 0.531441])


def test_rps(backend, to_backend, backend_kwargs):
    # test exceptions
    with pytest.raises(ValueError):
        sr.rps_score(to_backend(1), to_backend(1.1), **backend_kwargs)

    with pytest.raises(ValueError):
        sr.rps_score(to_backend(1), to_backend(-0.1), **backend_kwargs)

    with pytest.raises(ValueError):
        sr.rps_score(to_backend([1, 2]), to_backend(FCT_K), **backend_kwargs)

    # test correctness
    obs, fct = 3, [0.01, 0.32, 0.44, 0.23]
    res = sr.rps_score(to_backend(obs), to_backend(fct), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 0.1619)

    # onehot gives the same result
    res_onehot = sr.rps_score(
        to_backend([0, 0, 1, 0]), to_backend(fct), onehot=True, **backend_kwargs
    )
    assert_inferred(res_onehot, backend)
    assert np.isclose(np.asarray(res_onehot), 0.1619)

    # k_axis
    res1 = sr.rps_score(to_backend(OBS_K), to_backend(FCT_K), **backend_kwargs)
    res2 = sr.rps_score(
        to_backend(OBS_K), to_backend(np.transpose(FCT_K)), k_axis=0, **backend_kwargs
    )
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
        **backend_kwargs,
    )
    assert np.allclose(np.asarray(res1), np.asarray(res3))


def test_logs(backend, to_backend, backend_kwargs):
    # test exceptions
    with pytest.raises(ValueError):
        sr.log_score(to_backend(1), to_backend(1.1), **backend_kwargs)

    with pytest.raises(ValueError):
        sr.log_score(to_backend(1), to_backend(-0.1), **backend_kwargs)

    with pytest.raises(ValueError):
        sr.log_score(to_backend(0.5), to_backend(0.5), **backend_kwargs)

    # test correctness
    res = sr.log_score(to_backend(0), to_backend(0.481), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 0.6558514)

    res = sr.log_score(to_backend([0, 1]), to_backend([0.481, 0.481]))
    assert_inferred(res, backend)
    assert np.allclose(np.asarray(res), [0.6558514, -np.log(0.481)])


def test_rls(backend, to_backend, backend_kwargs):
    # test exceptions
    with pytest.raises(ValueError):
        sr.rls_score(to_backend(1), to_backend(1.1), **backend_kwargs)

    with pytest.raises(ValueError):
        sr.rls_score(to_backend(1), to_backend(-0.1), **backend_kwargs)

    # test correctness
    obs, fct = 2, [0.21, 0.31, 0.32, 0.16]
    res = sr.rls_score(to_backend(obs), to_backend(fct), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 1.064002)

    # k_axis
    res1 = sr.rls_score(to_backend(OBS_K), to_backend(FCT_K), **backend_kwargs)
    res2 = sr.rls_score(
        to_backend(OBS_K), to_backend(np.transpose(FCT_K)), k_axis=0, **backend_kwargs
    )
    assert_inferred(res1, backend)
    assert res1.shape == (4,)
    assert np.allclose(np.asarray(res1), np.asarray(res2))


def test_brier_family_float32(backend, to_backend_float32, backend_kwargs):
    obs_b = to_backend_float32([0, 1, 1])
    fct_b = to_backend_float32([0.2, 0.7, 0.4])
    obs_k = to_backend_float32(OBS_K)
    fct_k = to_backend_float32(FCT_K)
    for res in (
        sr.brier_score(obs_b, fct_b, **backend_kwargs),
        sr.log_score(obs_b, fct_b, **backend_kwargs),
        sr.rps_score(obs_k, fct_k, **backend_kwargs),
        sr.rls_score(obs_k, fct_k, **backend_kwargs),
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
    for fn, obs, fct in (
        (sr.brier_score, obs_b, [0.2, 0.7, 0.4]),
        (sr.log_score, obs_b, [0.2, 0.7, 0.4]),
        (sr.rps_score, obs_k, FCT_K),
        (sr.rls_score, obs_k, FCT_K),
    ):
        fct = torch.tensor(fct, dtype=torch.float64, requires_grad=True)
        fn(obs, fct).sum().backward()
        assert torch.isfinite(fct.grad).all(), fn.__name__
