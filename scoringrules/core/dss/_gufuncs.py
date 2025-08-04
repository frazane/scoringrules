import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_uv, lazy_gufunc_wrapper_mv


@lazy_gufunc_wrapper_uv
@guvectorize("(),(n),()->()")
def _dss_uv_gufunc(obs: np.ndarray, fct: np.ndarray, bias: bool, out: np.ndarray):
    ens_mean = np.mean(fct)
    correction = 0 if bias else 1
    sig = np.std(fct, ddof=correction)
    log_sig = 2 * np.log(sig)
    bias_precision = ((obs - ens_mean) / sig) ** 2
    out[0] = bias_precision + log_sig


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d),()->()")
def _dss_mv_gufunc(obs: np.ndarray, fct: np.ndarray, bias: bool, out: np.ndarray):
    M = fct.shape[0]
    ens_mean = np.sum(fct, axis=0) / M
    obs_cent = obs - ens_mean
    cov = np.cov(fct, rowvar=False, bias=bias)
    prec = np.linalg.inv(cov)
    log_det = np.log(np.linalg.det(cov))
    bias_precision = np.transpose(obs_cent) @ prec @ obs_cent
    out[0] = bias_precision + log_det
