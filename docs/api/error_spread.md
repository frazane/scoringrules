# Error Spread Score

The error spread score [(Christensen et al., 2015)](https://doi.org/10.1002/qj.2375) is given by:

$$ESS = \left(s^2 - e^2 - e \cdot s \cdot g\right)^2$$

where the mean $m$, variance $s^2$, and skewness $g$ of the ensemble forecast of size $F$ are computed as follows:

$$m = \frac{1}{F} \sum_{f=1}^{F} X_f, \quad s^2 = \frac{1}{F-1} \sum_{f=1}^{F} (X_f - m)^2, \quad g = \frac{F}{(F-1)(F-2)} \sum_{f=1}^{F} \left(\frac{X_f - m}{s}\right)^3$$

The error in the ensemble mean $e$ is calculated as $e = m - y$, where $y$ is the observed value.


::: scoringrules.error_spread_score
