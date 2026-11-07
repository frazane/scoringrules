"""Shared test helpers: gufunc/array-path parity and float32 inputs.

The numba gufuncs and the array-API path must give the same scores. Use the
``paths_agree`` fixture (``tests/conftest.py``), which skips when numba is not
selected, rather than importing ``assert_paths_agree`` directly.
"""

import numpy as np


def assert_paths_agree(fn, *args, rtol=1e-10, atol=1e-12, **kwargs):
    """Assert ``fn(*args, backend="numba", **kwargs)`` matches the inferred
    numpy path ``fn(*args, **kwargs)`` and that neither modifies its inputs.

    Both results go through ``np.asarray`` (lazy gufuncs return Python scalars
    for 0-d ``obs``). Returns the two results as numpy arrays.
    """
    inputs = [a for a in (*args, *kwargs.values()) if isinstance(a, np.ndarray)]
    snapshot = [a.copy() for a in inputs]
    res_numba = np.asarray(fn(*args, backend="numba", **kwargs))
    res_numpy = np.asarray(fn(*args, **kwargs))
    for before, after in zip(snapshot, inputs, strict=True):
        np.testing.assert_array_equal(after, before, "input modified in place")
    assert res_numba.shape == res_numpy.shape, (res_numba.shape, res_numpy.shape)
    assert res_numba.dtype.kind == "f", f"numba path returned {res_numba.dtype}"
    assert res_numpy.dtype.kind == "f", f"numpy path returned {res_numpy.dtype}"
    np.testing.assert_allclose(res_numba, res_numpy, rtol=rtol, atol=atol)
    return res_numba, res_numpy


def as_float32(x, backend):
    """``x`` as a float32 array of ``backend``'s framework (numba -> numpy)."""
    x = np.asarray(x, dtype=np.float32)
    if backend in ("numpy", "numba"):
        return x
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(x, dtype=jnp.float32)
    if backend == "torch":
        import torch

        return torch.as_tensor(x, dtype=torch.float32)
    raise ValueError(backend)


UV_CASES = ("default", "m_axis", "n1", "int_obs", "scalar_obs", "batch")


def uv_ensemble_inputs(case, M, weighted, seed=0, N=5):
    """``(obs, fct, kwargs)`` for a univariate ensemble parity case.

    ``case`` is one of ``UV_CASES``: ``"default"`` (``fct`` of shape ``(N, M)``),
    ``"m_axis"`` (``fct`` of shape ``(M, N)`` with ``m_axis=0``), ``"n1"``
    (``N=1``), ``"int_obs"`` (integer ``obs``), ``"scalar_obs"`` (a Python
    float ``obs`` with ``fct`` of shape ``(M,)``) and ``"batch"`` (an extra
    leading axis, ``fct`` of shape ``(2, N, M)``). With ``weighted``, ``kwargs``
    holds non-uniform ``ens_w`` laid out like ``fct``.
    """
    rng = np.random.default_rng(seed)
    batch = {"n1": (1,), "batch": (2, N)}.get(case, (N,))
    obs = rng.normal(size=batch)
    fct = rng.normal(size=(*batch, M))
    w = rng.uniform(0.1, 2.0, size=(*batch, M))
    kwargs = {}
    if case == "m_axis":
        fct, w = fct.T.copy(), w.T.copy()
        kwargs["m_axis"] = 0
    elif case == "int_obs":
        obs = rng.integers(-2, 3, size=batch)
    elif case == "scalar_obs":
        obs, fct, w = float(obs[0]), fct[0], w[0]
    if weighted:
        kwargs["ens_w"] = w
    return obs, fct, kwargs
