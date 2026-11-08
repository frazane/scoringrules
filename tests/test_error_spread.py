import numpy as np
import pytest
import scoringrules as sr
from ._parity import UV_CASES, uv_ensemble_inputs
from .conftest import assert_inferred


ENSEMBLE_SIZE = 11
N = 5


def test_error_spread_score(backend, to_backend, backend_kwargs):
    # test shape
    obs = np.random.randn(N)
    fct = obs[:, None] + np.random.randn(N, ENSEMBLE_SIZE)
    res = sr.error_spread_score(to_backend(obs), to_backend(fct), **backend_kwargs)
    assert_inferred(res, backend)
    assert res.shape == (N,)

    obs = np.random.randn(N)
    fct = np.random.randn(ENSEMBLE_SIZE, N)
    res = sr.error_spread_score(
        to_backend(obs), to_backend(fct), m_axis=0, **backend_kwargs
    )
    assert_inferred(res, backend)
    assert res.shape == (N,)

    # N=1 keeps the batch axis
    res = sr.error_spread_score(
        to_backend(obs[:1]), to_backend(fct.T[:1]), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert res.shape == (1,)

    # test correctness
    obs = 0.5
    fct = np.sqrt(np.arange(ENSEMBLE_SIZE))
    res = sr.error_spread_score(to_backend(obs), to_backend(fct), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isclose(np.asarray(res), 1.7942340)

    # approx zero when perfect forecast
    obs = np.random.randn(N)
    perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 1e-5
    res = sr.error_spread_score(
        to_backend(obs), to_backend(perfect_fct), **backend_kwargs
    )
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert not np.any(res - 0.0 > 0.0001)


# error_spread_score has no nan_policy: NaN inputs propagate, so of the nan cases
# only the propagate data applies (with the kwarg dropped).
ESS_CASES = [c for c in UV_CASES if not c.startswith("nan_")] + ["nan_propagate"]


@pytest.mark.parametrize("M", (6, 7))
@pytest.mark.parametrize("case", ESS_CASES)
def test_error_spread_score_paths_agree(case, M, paths_agree):
    obs, fct, kwargs = uv_ensemble_inputs(case, M, weighted=False)
    kwargs.pop("nan_policy", None)
    paths_agree(sr.error_spread_score, obs, fct, **kwargs)


def test_error_spread_score_float32(backend, to_backend_float32, backend_kwargs):
    obs = to_backend_float32(np.random.randn(N))
    fct = to_backend_float32(np.random.randn(N, ENSEMBLE_SIZE))
    for res in (
        sr.error_spread_score(obs, fct, **backend_kwargs),
        # Python-float obs take the forecast's dtype
        sr.error_spread_score([0.5] * N, fct, **backend_kwargs),
    ):
        assert_inferred(res, backend)
        assert res.dtype == fct.dtype


def test_error_spread_score_legacy_backend_string_deprecated():
    with pytest.warns(DeprecationWarning):
        sr.error_spread_score(
            np.random.randn(N), np.random.randn(N, ENSEMBLE_SIZE), backend="numpy"
        )


def test_error_spread_score_grad_torch():
    torch = pytest.importorskip("torch")
    obs = torch.tensor(np.random.randn(N), dtype=torch.float64)
    fct = torch.tensor(
        np.random.randn(N, ENSEMBLE_SIZE), dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(lambda f: sr.error_spread_score(obs, f), (fct,))


def test_error_spread_score_numba_dispatch(backend, to_backend):
    obs, fct = to_backend(np.random.randn(N)), to_backend(np.random.randn(N, 7))
    if backend in ("jax", "torch"):
        with pytest.raises(ValueError, match="numpy-compatible"):
            sr.error_spread_score(obs, fct, backend="numba")
    else:
        pytest.importorskip("numba")
        assert_inferred(sr.error_spread_score(obs, fct, backend="numba"), backend)


def test_error_spread_score_set_active_numba(numba_active, monkeypatch):
    from scoringrules.core import error_spread

    calls = []
    gufunc = error_spread._ess_gufunc

    def spy(*args):
        calls.append(args)
        return gufunc(*args)

    monkeypatch.setattr(error_spread, "_ess_gufunc", spy)
    obs, fct = np.random.randn(N), np.random.randn(N, ENSEMBLE_SIZE)
    sr.error_spread_score(obs, fct)
    assert len(calls) == 1
    torch = pytest.importorskip("torch")
    res = sr.error_spread_score(torch.as_tensor(obs), torch.as_tensor(fct))
    assert len(calls) == 1 and isinstance(res, torch.Tensor)
