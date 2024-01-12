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
sr.crps_lognormal(np.random.randn(100), np.random.uniform(0.5, 1.5, 100), np.random.lognormal(0, 1, 100))

# ensemble metrics
obs = np.random.randn(100)
fcts = obs[:,None] + np.random.randn(100, 21) * 0.1

sr.crps_ensemble(fcts, obs)
sr.error_spread_score(fcts, obs)

# multivariate ensemble metrics
obs = np.random.randn(100,3)
fcts = obs[:,None] + np.random.randn(100, 21, 3) * 0.1

sr.energy_score(fcts, obs)
sr.variogram_score(fcts, obs)
```

For the univariate ensemble metrics, the ensemble dimension is on the last axis unless you specify otherwise with the `axis` argument. For the multivariate ensemble metrics, the ensemble dimension and the variable dimension are on the second last and last axis respectively, unless specified otherwise with `m_axis` and `v_axis`.

## Backends
Scoringrules supports multiple backends. By default, the `numpy` and `numba` backends will be registered when importing the library. You can see the list of registered backends with

```python
print(sr.backends)
# {'numpy': <scoringrules.backend.numpy.NumpyBackend at 0x2ba2d6f391b0>,
# 'numba': <scoringrules.backend.numpy.NumbaBackend at 0x2ba2d6f38ac0>}
```

and the currently active backend, used by default in all metrics, can be seen with

```python
print(sr.backends.active)
# <scoringrules.backend.numpy.NumpyBackend at 0x2ba2d6f38ac0>
```

The default backend can also be changed with

```python
sr.backends.set_active("numba")
print(sr.backends.active)
# <scoringrules.backend.numpy.NumbaBackend at 0x2ba2d6f38ac0>
```
When computing a metric, the `backend` argument can be used to override the default choice.


To register a new backend, for example `torch`, simply use

```python
sr.register_backend("torch")
```

You can now use `torch` to compute metrics, either by setting it as the default backend or by specifying it on a specific metric:

```python
sr.crps_normal(0.1, 1.0, 0.0, backend="torch")
```
