# Special-function support audit

`scoringrules` needs a handful of special functions that the array-API standard
does not provide. The extension layer (`scoringrules/backend/extensions.py`)
dispatches **native-first** — using each framework's own implementation where it
exists (so autograd and device placement are preserved) — and falls back to
`scipy.special` only for plain numpy.

This document records, per function and per framework, whether a usable path
exists for the **forward** evaluation and for the **gradient**. The values below
were produced by probing the actually-installed frameworks (jax, torch, scipy)
on this branch, not from memory.

## Legend

- **PASS** — a usable native path exists. For jax this is `jax.scipy.special`;
  for torch it is either `torch.special` or a differentiable composition built
  from native primitives (`exp`/`lgamma`); for numpy it is `scipy.special`. The
  forward value is correct and, where the *grad* column says PASS, the gradient
  is finite.
- **PARTIAL (numpy round-trip)** — only reachable by detouring through
  numpy/scipy, which breaks the framework's native array type, autograd and
  non-CPU devices. Nothing in the layer uses this: there are no numpy
  round-trips for jax or torch (see also *Non-standard helpers* below).
- **BLOCKED** — no native and no differentiable path. The scipy fallback cannot
  be used because, on that framework, calling scipy raises (torch refuses
  `numpy()` on grad-requiring tensors; see probe output). These functions must
  receive numpy or jax inputs. The extension layer raises an explicit
  `NotImplementedError` for them on torch.

## Probe results (this branch)

```
JAX native:   erf T  gamma T  gammainc T  gammaincc T  beta T  betainc T  i0 T  i1 T  hyp2f1 T  expi T  factorial T  comb F
TORCH native: erf T  gamma F  gammainc T  gammaincc T  beta F  betainc F  i0 T  i1 T  hyp2f1 F  expi F  factorial F  comb F
torch scipy-fallback under grad: FAIL (RuntimeError: can't call numpy() on a tensor that requires grad)
jax   scipy-fallback under grad: PASS  (jax traces through scipy.special for these ufuncs)
```

Notes from the probe that shaped the implementation:

- `jax.scipy.special.hyp2f1` **exists and is correct** (the legacy backend had it
  commented out): `hyp2f1(1,1,2,0.5) = 1.386294`, matching scipy.
- `jax.scipy.special.expi` accepts **0-d input** fine (`expi(1.0) = 1.895118`),
  so no scalar-reshape workaround is needed.
- jax and torch have **no** `comb`. Composing it from factorials overflows (35!
  in float32, 171! in float64), so the layer works in log space:
  `exp(lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1))` (`jax.scipy.special.gammaln`,
  `torch.lgamma`), 0 outside `0 <= k <= n`, not rounded. numpy uses
  `scipy.special.comb` (`exact=False`), the legacy values.
- torch has no `gamma`/`beta`/`factorial`, but all three are built from the
  native, differentiable `torch.lgamma`, so they are PASS (forward + grad).

## Support matrix

| fn | numpy | jax fwd | jax grad | torch fwd | torch grad | notes |
|----|-------|---------|----------|-----------|------------|-------|
| `erf` | PASS | PASS | PASS | PASS | PASS | `jax.scipy.special.erf` / `torch.special.erf` |
| `gamma` | PASS | PASS | PASS | PASS | PASS | torch via `exp(lgamma)` |
| `gammainc` (reg. lower) | PASS | PASS | PASS | PASS | PASS | `torch.special.gammainc` |
| `gammalinc` (unreg. lower) | PASS | PASS | PASS | PASS | PASS | composed `gammainc·gamma` |
| `gammauinc` (unreg. upper) | PASS | PASS | PASS | PASS | PASS | composed `gammaincc·gamma`; `torch.special.gammaincc` present |
| `beta` | PASS | PASS | PASS | PASS | PASS | torch via `lgamma` |
| `betainc` (reg. incomplete) | PASS | PASS | PASS | BLOCKED | BLOCKED | no torch native; scipy raises under grad |
| `mbessel0` (`i0`) | PASS | PASS | PASS | PASS | PASS | `torch.special.i0` |
| `mbessel1` (`i1`) | PASS | PASS | PASS | PASS | PASS | `torch.special.i1` |
| `hypergeometric` (`hyp2f1`) | PASS | PASS | PASS | BLOCKED | BLOCKED | jax: `jax.scipy.special.hyp2f1`; torch: none |
| `expi` | PASS | PASS | PASS | BLOCKED | BLOCKED | no torch native; jax 0-d works |
| `factorial` | PASS | PASS | PASS | PASS | PASS | torch via `exp(lgamma(n+1))` |
| `comb` | PASS | PASS | PASS | PASS | PASS | `scipy.special.comb` (`exact=False`) for numpy; jax/torch via `exp` of `lgamma` differences (no overflow for large `n`, keeps float32, not rounded); grad w.r.t. real `n`, `k` |
| `quantile` | PASS | PASS | PASS | PASS | PASS | `np.quantile` / `jnp.quantile` / `torch.quantile` (linear interpolation); torch gets `q` as a tensor of `x`'s dtype and device; `axis` is an int or `None` |

`comb` is not rounded, so for integer arguments jax/torch return the exact value
up to rounding (about 1e-13 relative in float64). Unlike the old `round`ed
factorial form it is differentiable in `n` and `k`.

## Non-standard helpers

These are not special functions but are also provided by the extension layer:

- `apply_along_axis` — numpy/jax have native paths (jax via `vmap` with a
  `jnp.apply_along_axis` fallback). Torch has no `apply_along_axis`, so the
  layer builds one natively: it moves `axis` last, applies `torch.vmap(func1d)`
  over the flattened slices, and falls back to a Python loop over the slices
  (`torch.unbind` + `torch.stack`) when vmap cannot trace `func1d`
  (data-dependent control flow, `.item()`). Both paths keep autograd and the
  input's device.
- `cov` — `jnp.cov` / `torch.cov` (with `correction` mapped from `bias`) / `np.cov`.
- `indices` — `jnp.indices` / `np.indices` / torch `arange` + `meshgrid`.
  It takes `device=` (and `dtype=`), so pass `xp.device(x)` to create the grid
  next to the data.

Array creation (`arange`/`zeros`/`ones`/`full`/…) in the core passes
`device=xp.device(x)` and, for fractional values, `dtype=xp.float_dtype(x)`
taken from an input array. Otherwise torch creates on the default (CPU) device,
which fails for CUDA inputs, and numpy/torch fall back to their default float
dtype.
