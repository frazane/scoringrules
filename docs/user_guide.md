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

For numpy inputs, opt into the compiled [numba](https://numba.pydata.org/) gufuncs
with `backend="numba"`:

```python
sr.crps_ensemble(np.random.randn(5), np.random.randn(5, 11), backend="numba")
```

`backend="numba"` requires numpy-compatible inputs.

### Deprecations

Selecting an array-API backend explicitly is deprecated and will be removed in 1.0
— the framework is inferred from the input instead. This applies to:

- the `backend="numpy"`, `backend="jax"`, and `backend="torch"` arguments, and
- `sr.register_backend(...)` / `sr.backends.set_active(...)` for those backends.

`backend="numba"` is **not** deprecated; it remains the supported way to reach the
numba fast path.
