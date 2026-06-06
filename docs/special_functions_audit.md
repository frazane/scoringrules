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
  numpy/scipy, which breaks the framework's native array type and autograd. No
  special function currently needs this; the only numpy round-trip in the layer
  is the `apply_along_axis` helper on torch (see *Non-standard helpers* below).
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
- jax has **no** `comb`, and composing it as `factorial(n) // (factorial(k)·
  factorial(n-k))` is wrong: jax `factorial` is float32, and floor-division of
  `120.0001 // 12.00001` rounds to **9.0** instead of 10. The layer therefore
  composes `comb` with floating division and `round`, which gives 10.0 on numpy,
  jax, and torch.
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
| `comb` | PASS | PASS | n/a | PASS | n/a | composed from `factorial` with `/ + round`; discrete, so gradient is not meaningful |

`n/a` for `comb` grad: `comb` is integer-valued and routed through `round`, so it
has no meaningful gradient (this matched the old `//` form, which was also
non-differentiable). `factorial` itself, built from the smooth `lgamma`, *is*
differentiable.

## Non-standard helpers

These are not special functions but are also provided by the extension layer:

- `apply_along_axis` — numpy/jax have native paths (jax via `vmap` with a
  `jnp.apply_along_axis` fallback); **torch has no native equivalent**, so it is
  the one **numpy round-trip** in the layer (PARTIAL): the per-slice callable is
  evaluated through `numpy.apply_along_axis` and the result re-wrapped as a
  tensor. This breaks autograd on torch and should be avoided in grad-sensitive
  code paths; it also requires CPU tensors, since `numpy()` raises on CUDA
  tensors.
- `cov` — `jnp.cov` / `torch.cov` (with `correction` mapped from `bias`) / `np.cov`.
- `indices` — `jnp.indices` / `np.indices` (torch wraps `np.indices`).
