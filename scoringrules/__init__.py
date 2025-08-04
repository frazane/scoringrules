from importlib.metadata import version
import functools
import warnings

from scoringrules._brier import (
    brier_score,
    rps_score,
    log_score,
    rls_score,
)
from scoringrules._crps import (
    crps_beta,
    crps_binomial,
    crps_ensemble,
    crps_exponential,
    crps_exponentialM,
    crps_2pexponential,
    crps_gamma,
    crps_gev,
    crps_gpd,
    crps_gtclogistic,
    crps_tlogistic,
    crps_clogistic,
    crps_gtct,
    crps_tt,
    crps_ct,
    crps_gtcnormal,
    crps_tnormal,
    crps_cnormal,
    crps_hypergeometric,
    crps_laplace,
    crps_logistic,
    crps_loglaplace,
    crps_loglogistic,
    crps_lognormal,
    crps_mixnorm,
    crps_negbinom,
    crps_normal,
    crps_2pnormal,
    crps_poisson,
    crps_t,
    crps_uniform,
    crps_quantile,
    owcrps_ensemble,
    twcrps_ensemble,
    vrcrps_ensemble,
)
from scoringrules._energy import (
    es_ensemble,
    owes_ensemble,
    twes_ensemble,
    vres_ensemble,
)
from scoringrules._dss import dss_ensemble
from scoringrules._error_spread import error_spread_score
from scoringrules._interval import interval_score, weighted_interval_score
from scoringrules._logs import (
    logs_beta,
    logs_binomial,
    logs_ensemble,
    logs_exponential,
    logs_exponential2,
    logs_2pexponential,
    logs_gamma,
    logs_gev,
    logs_gpd,
    logs_hypergeometric,
    logs_laplace,
    logs_loglaplace,
    logs_logistic,
    logs_loglogistic,
    logs_lognormal,
    logs_mixnorm,
    logs_negbinom,
    logs_normal,
    logs_2pnormal,
    logs_poisson,
    logs_t,
    logs_tlogistic,
    logs_tnormal,
    logs_tt,
    logs_uniform,
    clogs_ensemble,
)
from scoringrules._variogram import (
    owvs_ensemble,
    twvs_ensemble,
    vs_ensemble,
    vrvs_ensemble,
)
from scoringrules._kernels import (
    gksuv_ensemble,
    twgksuv_ensemble,
    owgksuv_ensemble,
    vrgksuv_ensemble,
    gksmv_ensemble,
    twgksmv_ensemble,
    owgksmv_ensemble,
    vrgksmv_ensemble,
)
from scoringrules._quantile import quantile_score
from scoringrules.backend import backends, register_backend


def _deprecated_alias(new_func, old_func_name, remove_version):
    """Return a deprecated alias for a renamed function."""

    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_func_name} is deprecated and will be removed in {remove_version}. "
            f"Use {new_func.__name__} instead.",
            DeprecationWarning,
        )
        return new_func(*args, **kwargs)

    return wrapper


energy_score = _deprecated_alias(es_ensemble, "energy_score", "v1.0.0")
owenergy_score = _deprecated_alias(owes_ensemble, "owenergy_score", "v1.0.0")
twenergy_score = _deprecated_alias(twes_ensemble, "twenergy_score", "v1.0.0")
vrenergy_score = _deprecated_alias(vres_ensemble, "vrenergy_score", "v1.0.0")

variogram_score = _deprecated_alias(vs_ensemble, "variogram_score", "v1.0.0")
owvariogram_score = _deprecated_alias(owvs_ensemble, "owvariogram_score", "v1.0.0")
twvariogram_score = _deprecated_alias(twvs_ensemble, "twvariogram_score", "v1.0.0")
vrvariogram_score = _deprecated_alias(vrvs_ensemble, "vrvariogram_score", "v1.0.0")

__version__ = version("scoringrules")

__all__ = [
    "register_backend",
    "backends",
    "crps_ensemble",
    "crps_beta",
    "crps_binomial",
    "crps_exponential",
    "crps_exponentialM",
    "crps_2pexponential",
    "crps_gamma",
    "crps_gev",
    "crps_gpd",
    "crps_gtclogistic",
    "crps_tlogistic",
    "crps_clogistic",
    "crps_gtcnormal",
    "crps_tnormal",
    "crps_cnormal",
    "crps_gtct",
    "crps_tt",
    "crps_ct",
    "crps_hypergeometric",
    "crps_laplace",
    "crps_logistic",
    "crps_loglaplace",
    "crps_loglogistic",
    "crps_lognormal",
    "crps_mixnorm",
    "crps_negbinom",
    "crps_normal",
    "crps_2pnormal",
    "crps_poisson",
    "crps_quantile",
    "crps_t",
    "crps_uniform",
    "owcrps_ensemble",
    "twcrps_ensemble",
    "vrcrps_ensemble",
    "logs_beta",
    "logs_binomial",
    "logs_ensemble",
    "logs_exponential",
    "logs_exponential2",
    "logs_2pexponential",
    "logs_gamma",
    "logs_gev",
    "logs_gpd",
    "logs_hypergeometric",
    "logs_laplace",
    "logs_loglaplace",
    "logs_logistic",
    "logs_loglogistic",
    "logs_lognormal",
    "logs_mixnorm",
    "logs_negbinom",
    "logs_normal",
    "logs_2pnormal",
    "logs_poisson",
    "logs_t",
    "logs_tlogistic",
    "logs_tnormal",
    "logs_tt",
    "logs_uniform",
    "clogs_ensemble",
    "brier_score",
    "rps_score",
    "log_score",
    "rls_score",
    "dss_ensemble",
    "error_spread_score",
    "es_ensemble",
    "owes_ensemble",
    "twes_ensemble",
    "vres_ensemble",
    "vs_ensemble",
    "owvs_ensemble",
    "twvs_ensemble",
    "vrvs_ensemble",
    "gksuv_ensemble",
    "twgksuv_ensemble",
    "owgksuv_ensemble",
    "vrgksuv_ensemble",
    "gksmv_ensemble",
    "twgksmv_ensemble",
    "owgksmv_ensemble",
    "vrgksmv_ensemble",
    "interval_score",
    "weighted_interval_score",
    "quantile_score",
]
