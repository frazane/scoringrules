# Scoring rules for categorical outcomes

Suppose that the outcome $Y \in \{1, 2, \dots, K\}$ is one of $K$ possible categories.
Then, a probabilistic forecast $F$ for $Y$ is a vector $F = (F_{1}, \dots, F_{K})$
with $\sum_{i=1}^{K} F_{i} = 1$, containing the forecast probabilities that $Y = 1, \dots, Y = K$.

When $K = 2$, it is common to instead consider a binary outcome $Y \in \{0, 1\}$, which
represents an event that either occurs $(Y = 1)$ or does not $(Y = 0)$. The forecast in this
case is typically represented by a single probability $F \in [0, 1]$ that $Y = 1$, rather than
the vector $(F, 1 - F)$. However, evaluating these probability forecasts is a particular case of
evaluation methods for the more general categorical case described above.


## Brier Score

::: scoringrules.brier_score

## Log Score

::: scoringrules.log_score

## Ranked Probability Score

::: scoringrules.rps_score

## Ranked Log Score

::: scoringrules.rls_score
