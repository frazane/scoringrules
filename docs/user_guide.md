# User guide

## First steps
Start by importing the library in your code with
```python
import scoringrules as sr
```

the library API is simple: all metrics are available under the main namespace. Let's look at some examples:

```python
import numpy as np

# on scalars
sr.brier_score(0.1, 1)
sr.crps_normal(0.1, 1.2, 0.3)

# on arrays
sr.brier_score(np.random.uniform(0, 1, 100), np.random.binomial(1, 0.5, 100))
sr.crps_lognormal(np.random.lognormal(0, 1, 100), np.random.randn(100), np.random.uniform(0.5, 1.5, 100))

# ensemble metrics
obs = np.random.randn(100)
fct = obs[:,None] + np.random.randn(100, 21) * 0.1

sr.crps_ensemble(obs, fct)
sr.error_spread_score(obs, fct)

# multivariate ensemble metrics
obs = np.random.randn(100,3)
fct = obs[:,None] + np.random.randn(100, 21, 3) * 0.1

sr.energy_score(obs, fct)
sr.variogram_score(obs, fct)
```

For the univariate ensemble metrics, the ensemble dimension is on the last axis unless you specify otherwise with the `axis` argument. For the multivariate ensemble metrics, the ensemble dimension and the variable dimension are on the second last and last axis respectively, unless specified otherwise with `m_axis` and `v_axis`.

## Backends

Scoringrules runs every score across multiple array frameworks — numpy, jax, and
torch — from a single implementation. The framework is **inferred from the input
arrays**: pass numpy, jax, or torch arrays and the result is returned in the same
framework, with no configuration required.

```python
import numpy as np
import scoringrules as sr

sr.crps_normal(np.array([0.1]), np.array([0.0]), np.array([1.0]))  # numpy array in, numpy array out
```

```python
import jax.numpy as jnp

sr.crps_normal(jnp.array([0.1]), jnp.array([0.0]), jnp.array([1.0]))  # jax array out
```

```python
import torch

mu = torch.tensor([0.0], requires_grad=True)
sr.crps_normal(torch.tensor([0.1]), mu, torch.tensor([1.0]))  # torch tensor out; autograd preserved
```

Inputs must come from a single framework; mixing (e.g. a numpy observation with a
torch forecast) raises an error.

### The numba fast path

For numpy inputs, scoringrules uses compiled [numba](https://numba.pydata.org/)
gufuncs when numba is installed (`pip install scoringrules[numba]`). Nothing needs to
be configured:

```python
sr.crps_ensemble(np.random.randn(5), np.random.randn(5, 11))  # numba gufunc if installed
```

Some inputs always take the array path, with or without numba: jax and torch arrays,
masked arrays, bool, float16, longdouble and complex data, and objects that only
provide `__array__`. Scores that have no gufunc always take it too. In the default
mode a few combinations also stay on the array path because it is faster or more
accurate there. At the moment these are `weighted_interval_score`, and the `akr` and `akr_circperm`
estimators of `gksuv_ensemble` for float32 data.

Use `set_config` to change the mode for the whole process:

```python
sr.set_config(numba=False)    # never use numba
sr.set_config(numba=True)     # always use a gufunc when the score has one
sr.set_config(numba="auto")   # the default
sr.get_config()               # {'numba': 'auto'}
```

With `numba=True`, a score call that would use a gufunc raises an `ImportError` if
numba is not installed (inputs that always take the array path still work). The
excluded combinations above then use the gufunc as well. To change the mode for a block of code only, use `config_context`. It restores the
previous value on exit and does not affect other threads:

```python
with sr.config_context(numba=False):
    sr.crps_ensemble(obs, fct)  # array path
```

Work that runs in other threads, such as a thread pool or dask's threaded scheduler,
does not see a `config_context` opened in the calling thread; use `set_config` there.
Spawned processes only read the environment variable below.

The initial mode is read from the environment variable `SCORINGRULES_NUMBA` when
scoringrules is imported. It accepts `0`, `1` and `auto` (also `false` and `true`).

Compiling a gufunc takes about 0.1 to 1.5 seconds per score the first time it runs on
a machine. The compiled code is then cached on disk, and later processes only pay a
few milliseconds. If the installation directory is read-only, set `NUMBA_CACHE_DIR`
to a writable directory before the first call. Importing scoringrules does not import
numba. The first score call that uses a gufunc does, which takes about 0.2 seconds.

### Deprecations

Selecting a backend explicitly is deprecated and will be removed in 1.0, since the
array framework is inferred from the input and numba is controlled by `set_config`.
This applies to:

- every value of the `backend` argument of the scores. `backend="numba"` still works
  as `numba=True` for that call, the other values are ignored. All of them warn.
- `sr.register_backend(...)` and `sr.backends.set_active(...)`.
  `set_active("numba")` is the same as `set_config(numba=True)`,
  `set_active("numpy")` is the same as `set_config(numba=False)`, and `"jax"` or
  `"torch"` are the same as `set_config(numba="auto")`.
- reading values from the `sr.backends` registry: any `sr.backends[...]` lookup,
  `sr.backends.get(...)`, `.values()`, `.items()` and `.active` warn. It is a
  compatibility shim: its values are the array-API namespaces scoringrules uses
  internally, so standard array-API calls such as `sr.backends["torch"].sum(x, axis=-1)`
  keep working until 1.0. Checking for a backend (`"torch" in sr.backends`,
  `sr.backends.available_backends`) does not warn.
