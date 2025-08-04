# API reference

This page provides a summary of scoringrules' API. All functions are
available in the top-level namespace of the package and are organised here by category.

```{eval-rst}
.. currentmodule:: scoringrules

Categorical forecasts
=====================
.. autosummary::
    :toctree: generated

    brier_score
    rps_score
    log_score
    rls_score


Consistent scoring functions
============================
.. autosummary::
    :toctree: generated

    quantile_score
    interval_score
    weighted_interval_score


Scoring rules for ensemble (sample) forecasts
==================

Univariate
------------------

.. autosummary::
    :toctree: generated

    crps_ensemble
    logs_ensemble
    gksuv_ensemble

Multivariate
--------------------

.. autosummary::
    :toctree: generated

    es_ensemble
    vs_ensemble
    gksmv_ensemble

Weighted scoring rules
--------------------

.. autosummary::
    :toctree: generated

    twcrps_ensemble
    owcrps_ensemble
    vrcrps_ensemble

    clogs_ensemble

    twgksuv_ensemble
    owgksuv_ensemble
    vrgksuv_ensemble

    owes_ensemble
    twes_ensemble
    vres_ensemble

    owvs_ensemble
    twvs_ensemble
    vrvs_ensemble

    twgksmv_ensemble
    owgksmv_ensemble
    vrgksmv_ensemble


Scoring rules for parametric forecast distributions
====================================

CRPS
--------------------

.. autosummary::
    :toctree: generated

    crps_beta
    crps_binomial
    crps_exponential
    crps_exponentialM
    crps_2pexponential
    crps_gamma
    crps_gev
    crps_gpd
    crps_gtclogistic
    crps_tlogistic
    crps_clogistic
    crps_gtcnormal
    crps_tnormal
    crps_cnormal
    crps_gtct
    crps_tt
    crps_ct
    crps_hypergeometric
    crps_laplace
    crps_logistic
    crps_loglaplace
    crps_loglogistic
    crps_lognormal
    crps_mixnorm
    crps_negbinom
    crps_normal
    crps_2pnormal
    crps_poisson
    crps_quantile
    crps_t
    crps_uniform


Log score
--------------------

.. autosummary::
    :toctree: generated

    logs_beta
    logs_binomial
    logs_ensemble
    logs_exponential
    logs_exponential2
    logs_2pexponential
    logs_gamma
    logs_gev
    logs_gpd
    logs_hypergeometric
    logs_laplace
    logs_loglaplace
    logs_logistic
    logs_loglogistic
    logs_lognormal
    logs_mixnorm
    logs_negbinom
    logs_normal
    logs_2pnormal
    logs_poisson
    logs_t
    logs_tlogistic
    logs_tnormal
    logs_tt
    logs_uniform


Backends
========
.. autosummary::
    :toctree: generated

    register_backend
```
