# API reference

This page provides a summary of scoringrules' API. All functions are
available in the top-level namespace of the package and are here
organized by category.

```{eval-rst}
.. currentmodule:: scoringrules

Ensemble forecasts
==================

Univariate
------------------

.. autosummary::
    :toctree: generated

    crps_ensemble
    twcrps_ensemble
    owcrps_ensemble
    vrcrps_ensemble
    gksuv_ensemble
    twgksuv_ensemble
    owgksuv_ensemble
    vrgksuv_ensemble
    <!-- clogs_ensemble -->
    error_spread_score

.. Multivariate
.. --------------------

.. .. autosummary::
..     :toctree: generated

..     energy_score
..     twenergy_score
..     owenergy_score
..     vrenergy_score
..     variogram_score
..     owvariogram_score
..     twvariogram_score
..     vrvariogram_score
..     gksmv_ensemble
..     twgksmv_ensemble
..     owgksmv_ensemble
..     vrgksmv_ensemble

.. Parametric distributions forecasts
.. ====================================
.. .. autosummary::
..     :toctree: generated

..     crps_beta
..     crps_binomial
..     crps_exponential
..     crps_exponentialM
..     crps_2pexponential
..     crps_gamma
..     crps_gev
..     crps_gpd
..     crps_gtclogistic
..     crps_tlogistic
..     crps_clogistic
..     crps_gtcnormal
..     crps_tnormal
..     crps_cnormal
..     crps_gtct
..     crps_tt
..     crps_ct
..     crps_hypergeometric
..     crps_laplace
..     crps_logistic
..     crps_loglaplace
..     crps_loglogistic
..     crps_lognormal
..     crps_mixnorm
..     crps_negbinom
..     crps_normal
..     crps_2pnormal
..     crps_poisson
..     crps_quantile
..     crps_t
..     crps_uniform
..     logs_beta
..     logs_binomial
..     logs_ensemble
..     logs_exponential
..     logs_exponential2
..     logs_2pexponential
..     logs_gamma
..     logs_gev
..     logs_gpd
..     logs_hypergeometric
..     logs_laplace
..     logs_loglaplace
..     logs_logistic
..     logs_loglogistic
..     logs_lognormal
..     logs_mixnorm
..     logs_negbinom
..     logs_normal
..     logs_2pnormal
..     logs_poisson
..     logs_t
..     logs_tlogistic
..     logs_tnormal
..     logs_tt
..     logs_uniform

.. Functions of distributions
.. ==========================
.. .. autosummary::
..     :toctree: generated

..     interval_score
..     weighted_interval_score


.. Categorical forecasts
.. =================================
.. .. autosummary::
..     :toctree: generated

..     brier_score
..     rps_score
..     log_score
..     rls_score


.. Backends
.. ========
.. .. autosummary::
..     :toctree: generated

..     register_backend
```
