import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:])",
        "void(float64[:], float64[:,:], float64[:])",
    ],
    "(d),(m,d)->()",
)
def _dss_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    M = fct.shape[0]
    bias = obs - (np.sum(fct, axis=0) / M)
    cov = np.cov(fct, rowvar=False).astype(bias.dtype)
    prec = np.linalg.inv(cov).astype(bias.dtype)
    log_det = np.log(np.linalg.det(cov))
    bias_precision = bias.T @ prec @ bias
    out[0] = log_det + bias_precision
