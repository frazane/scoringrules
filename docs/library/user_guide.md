# User guide

## First steps
Start by importing the library using
```python
import scoringrules as sr
```

The library API is simple: all metrics are available under the main namespace. Some examples are given below.

```python
import numpy as np

## scalar data

# Brier score (observation = 1, forecast = 0.1)
sr.brier_score(1, 0.1)

# CRPS (observation = 0.1, forecast = normal distribution with mean 1.2 and sd 0.3)
sr.crps_normal(0.1, 1.2, 0.3)


## array data

# Brier score
obs = np.random.uniform(0, 1, 100)
fct = np.random.binomial(1, 0.5, 100)
sr.brier_score(obs, fct)

# CRPS (forecast = lognormal distribution)
obs = np.random.lognormal(0, 1, 100)
mu = np.random.randn(100)
sig = np.random.uniform(0.5, 1.5, 100)
sr.crps_lognormal(obs, mu, sig)


## ensemble metrics (univariate)
obs = np.random.randn(100)
fct = obs[:, None] + np.random.randn(100, 21) * 0.1

sr.crps_ensemble(obs, fct) # CRPS
sr.gksuv_ensemble(obs, fct) # univariate Gaussian kernel score


## ensemble metrics (multivariate)
obs = np.random.randn(100, 3)
fct = obs[:, None] + np.random.randn(100, 21, 3) * 0.1

sr.energy_score(obs, fct) # energy score
sr.variogram_score(obs, fct) # variogram score
```

For the ensemble metrics, the forecast should have one more dimension than the observations, containing the ensemble (i.e. sample) members. In the univariate case, this ensemble dimension is assumed to be the last axis of the input forecast array, though this can be changed manually by specifying the `m_axis` argument (default is `m_axis=-1`). In the multivariate case, the ensemble dimension is assumed to be the second last axis, with the number of variables the last axis. These can similarly be changed manually by specifying the `m_axis` and `v_axis` arguments respectively (default is `m_axis=-2` and `v_axis=-1`).

## Backends

The scoringrules library supports multiple backends: `numpy` (accelerated with `numba`), `torch`, `tensorflow`, and `jax`. By default, the `numpy` and `numba` backends will be registered when importing the library. You can see the list of registered backends by running

```python
print(sr.backends)
# {'numpy': <scoringrules.backend.numpy.NumpyBackend at 0x2ba2d6f391b0>,
# 'numba': <scoringrules.backend.numpy.NumbaBackend at 0x2ba2d6f38ac0>}
```

and the currently active backend, used by default in all metrics, by running

```python
print(sr.backends.active)
# <scoringrules.backend.numpy.NumpyBackend at 0x2ba2d6f38ac0>
```

The default backend can be changed manually using `sr.backends.set_active()`. For example, the following code sets the active backend to `numba`.

```python
sr.backends.set_active("numba")
print(sr.backends.active)
# <scoringrules.backend.numpy.NumbaBackend at 0x2ba2d6f38ac0>
```
Alternatively, the `backend` argument to the score functions can be used to override the default choice. For example,

```python
sr.crps_normal(0.1, 1.2, 0.3, backend="numba")
```

To register a new backend, for example `torch`, simply use

```python
sr.register_backend("torch")
```

You can now use `torch` to compute scores, either by setting it as the default backend, or by specifying it manually when running the desired function:

```python
sr.crps_normal(0.1, 1.2, 0.3, backend="torch")
```
