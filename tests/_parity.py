"""Shared test helpers: gufunc/array-path parity and float32 inputs.

The numba gufuncs and the array-API path must give the same scores. Use the
``paths_agree`` fixture (``tests/conftest.py``), which skips when numba is not
selected, rather than importing ``assert_paths_agree`` directly.
"""

import numpy as np
import pytest


def assert_paths_agree(
    fn,
    *args,
    rtol=1e-10,
    atol=1e-12,
    check_dtype=True,
    allow_all_nan=False,
    raises=None,
    **kwargs,
):
    """Assert ``fn(*args, backend="numba", **kwargs)`` matches the inferred
    numpy path ``fn(*args, **kwargs)`` and that neither modifies its inputs.

    Both results go through ``np.asarray`` (lazy gufuncs return numpy scalars
    for 0-d ``obs``) and must have the same shape, the same dtype (unless
    ``check_dtype=False``, for families where only shapes and values matter)
    and agree within ``rtol``/``atol``, NaN positions included. An all-NaN
    result fails unless ``allow_all_nan=True``, so that two paths that both
    break cannot agree by accident. With ``raises``, both paths must raise
    that exception instead. Returns the two results as numpy arrays.
    """
    inputs = [a for a in (*args, *kwargs.values()) if isinstance(a, np.ndarray)]
    snapshot = [a.copy() for a in inputs]

    def check_unchanged(path):
        for before, after in zip(snapshot, inputs, strict=True):
            np.testing.assert_array_equal(
                after, before, f"{path} path modified an input in place"
            )

    if raises is not None:
        with pytest.raises(raises):
            fn(*args, backend="numba", **kwargs)
        check_unchanged("numba")
        with pytest.raises(raises):
            fn(*args, **kwargs)
        check_unchanged("numpy")
        return None
    res_numba = np.asarray(fn(*args, backend="numba", **kwargs))
    check_unchanged("numba")
    res_numpy = np.asarray(fn(*args, **kwargs))
    check_unchanged("numpy")
    assert res_numba.shape == res_numpy.shape, (res_numba.shape, res_numpy.shape)
    assert res_numba.dtype.kind == "f", f"numba path returned {res_numba.dtype}"
    assert res_numpy.dtype.kind == "f", f"numpy path returned {res_numpy.dtype}"
    if check_dtype:
        assert res_numba.dtype == res_numpy.dtype, (res_numba.dtype, res_numpy.dtype)
    if not allow_all_nan:
        assert not np.all(np.isnan(res_numpy)), "all-NaN result"
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


UV_CASES = (
    "default",
    "m_axis",
    "n1",
    "int_obs",
    "scalar_obs",
    "batch",
    "list_inputs",
    "int_fct",
    "nan_propagate",
    "nan_omit",
    "nan_raise",
    "nan_omit_m_axis",
)


def uv_ensemble_inputs(case, M, weighted, seed=0, N=5):
    """``(obs, fct, kwargs)`` for a univariate ensemble parity case.

    ``case`` is one of ``UV_CASES``:

    - ``"default"``: ``fct`` of shape ``(N, M)``;
    - ``"m_axis"``: ``fct`` of shape ``(M, N)`` with ``m_axis=0``;
    - ``"n1"``: ``N=1``;
    - ``"int_obs"`` / ``"int_fct"``: integer ``obs`` / integer ``fct`` (with ties);
    - ``"scalar_obs"``: a Python float ``obs`` with ``fct`` of shape ``(M,)``;
    - ``"batch"``: an extra leading axis, ``fct`` of shape ``(2, N, M)``;
    - ``"list_inputs"``: ``obs``, ``fct`` (and ``ens_w``) as nested lists;
    - ``"nan_<policy>"``: NaN members in row 0, a NaN ``obs`` in row 4 and (when
      weighted) a NaN weight in row 2, with ``nan_policy=<policy>``;
      ``"nan_omit_m_axis"`` is ``"nan_omit"`` laid out as ``"m_axis"``.

    With ``weighted``, ``kwargs`` holds non-uniform ``ens_w`` laid out like ``fct``.
    """
    rng = np.random.default_rng(seed)
    batch = {"n1": (1,), "batch": (2, N)}.get(case, (N,))
    obs = rng.normal(size=batch)
    fct = rng.normal(size=(*batch, M))
    w = rng.uniform(0.1, 2.0, size=(*batch, M))
    kwargs = {}
    if case.startswith("nan_"):
        fct[0, [1, 3]] = np.nan
        obs[4] = np.nan
        w[2, 0] = np.nan
        kwargs["nan_policy"] = case.split("_")[1]
    if case in ("m_axis", "nan_omit_m_axis"):
        fct, w = fct.T.copy(), w.T.copy()
        kwargs["m_axis"] = 0
    elif case == "int_obs":
        obs = rng.integers(-2, 3, size=batch)
    elif case == "int_fct":
        fct = rng.integers(-3, 4, size=fct.shape)
    elif case == "scalar_obs":
        obs, fct, w = float(obs[0]), fct[0], w[0]
    elif case == "list_inputs":
        obs, fct, w = obs.tolist(), fct.tolist(), w.tolist()
    if weighted:
        kwargs["ens_w"] = w
    return obs, fct, kwargs


MV_CASES = ("default", "m_axis", "v_axis", "n1", "int_obs", "single", "batch")


def mv_ensemble_inputs(case, M, weighted, seed=0, N=5, D=3):
    """``(obs, fct, kwargs)`` for a multivariate ensemble parity case.

    ``case`` is one of ``MV_CASES``: ``"default"`` (``obs`` of shape ``(N, D)``,
    ``fct`` of shape ``(N, M, D)``), ``"m_axis"`` (``fct`` of shape ``(M, N, D)``
    with ``m_axis=0``), ``"v_axis"`` (``fct`` of shape ``(N, D, M)`` with
    ``m_axis=-1, v_axis=-2``), ``"n1"`` (``N=1``), ``"int_obs"`` (integer
    ``obs``), ``"single"`` (``obs`` of shape ``(D,)``, ``fct`` of shape
    ``(M, D)``) and ``"batch"`` (an extra leading axis, ``fct`` of shape
    ``(2, N, M, D)``). With ``weighted``, ``kwargs`` holds non-uniform ``ens_w``
    laid out like ``fct`` without its variables axis.
    """
    rng = np.random.default_rng(seed)
    batch = {"n1": (1,), "single": (), "batch": (2, N)}.get(case, (N,))
    obs = rng.normal(size=(*batch, D))
    fct = rng.normal(size=(*batch, M, D))
    w = rng.uniform(0.1, 2.0, size=(*batch, M))
    kwargs = {}
    if case == "m_axis":
        fct, w = np.moveaxis(fct, -2, 0).copy(), np.moveaxis(w, -1, 0).copy()
        kwargs["m_axis"] = 0
    elif case == "v_axis":
        fct = np.swapaxes(fct, -1, -2).copy()
        kwargs["m_axis"], kwargs["v_axis"] = -1, -2
    elif case == "int_obs":
        obs = rng.integers(-2, 3, size=obs.shape)
    if weighted:
        kwargs["ens_w"] = w
    return obs, fct, kwargs


def mark_known_mv_failures(request, case, weighted):
    """Apply the strict xfails shared by every multivariate parity test."""
    if case == "v_axis" and weighted:
        # Both paths fail the same way, so this is not a parity issue. Here the
        # moved axis is out of range and it raises; when it happens to be in range
        # (e.g. M == N) the wrong ens_w axis is used silently.
        request.applymarker(
            pytest.mark.xfail(
                raises=ValueError,
                strict=True,
                reason="ens_w's m_axis is taken relative to fct.ndim, so with v_axis "
                "before m_axis ens_w raises or silently uses the wrong weights "
                "(apply_nan_policy_ens_mv, core/utils.py:243, for the energy scores; "
                "multivariate_weight_check, core/utils.py:117, for the others); "
                "not filed, see PR 1 notes",
            )
        )


def mv_w_func(x):
    """A smooth, positive multivariate weight function (numpy only)."""
    return 1.0 / (1.0 + np.exp(-np.mean(x)))


def mv_v_func(x):
    """A multivariate chaining function (numpy only)."""
    return np.maximum(x, -0.5)


def uv_w_func(x):
    """A smooth, positive univariate weight function (numpy only)."""
    return 1.0 / (1.0 + np.exp(-x))
