import numpy as np
import pytest

import scoringrules as sr
from .conftest import assert_inferred


def test_quantile_score(backend, to_backend, backend_kwargs):
    obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    fct = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    res = sr.quantile_score(
        to_backend(obs), to_backend(fct), to_backend(alpha), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert res.shape == (5,)
    assert np.allclose(np.asarray(res), 0.0)

    # correctness
    fa = to_backend
    res = sr.quantile_score(fa(0.3), fa(0.5), fa(0.2), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 0.16)
    res = sr.quantile_score(fa(0.3), fa(0.1), fa(0.2), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 0.04)

    # broadcasting a vector of levels against a batch of forecasts
    obs = np.array([0.0, 1.0])
    fct = np.array([[-1.0, 0.5], [2.0, 0.0]])
    alpha = np.array([0.25, 0.75])
    res = sr.quantile_score(fa(obs[:, None]), fa(fct), fa(alpha), **backend_kwargs)
    assert_inferred(res, backend)
    expected = np.where(obs[:, None] < fct, 1 - alpha, -alpha) * (fct - obs[:, None])
    assert np.allclose(np.asarray(res), expected)

    # value checks
    with pytest.raises(ValueError):
        sr.quantile_score(fa(0.3), fa(0.5), fa(-0.2), **backend_kwargs)
    with pytest.raises(ValueError):
        sr.quantile_score(fa(0.3), fa(0.5), fa([0.2, 1.2]), **backend_kwargs)


def test_quantile_score_float32(backend, to_backend_float32, backend_kwargs):
    obs = to_backend_float32(np.random.randn(5))
    fct = to_backend_float32(np.random.randn(5))
    alpha = to_backend_float32(np.full(5, 0.3))
    res = sr.quantile_score(obs, fct, alpha, **backend_kwargs)
    assert_inferred(res, backend)
    assert res.dtype == fct.dtype


def test_quantile_score_legacy_backend_string_deprecated():
    with pytest.warns(DeprecationWarning):
        sr.quantile_score(0.3, 0.5, 0.2, backend="numpy")


def test_quantile_score_grad_torch():
    torch = pytest.importorskip("torch")
    obs = torch.tensor([0.3, 0.6], dtype=torch.float64)
    fct = torch.tensor([0.5, 0.1], dtype=torch.float64, requires_grad=True)
    sr.quantile_score(obs, fct, 0.2).sum().backward()
    # d/dfct = 1{obs < fct} - alpha
    assert torch.allclose(fct.grad, torch.tensor([0.8, -0.2], dtype=torch.float64))
