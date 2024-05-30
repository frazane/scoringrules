# Interval Score

## Interval or Winkler Score

For a prediction interval (PI), the interval or Winkler score is given by:

$\text{IS} = \begin{cases}
    (u - l) + 2\frac{2}{\alpha}(l - y)  & \text{for } y < l \\
    (u - l)                             & \text{for } l \leq y \leq u \\
    (u - l) + \frac{2}{\alpha}(y - u)   & \text{for } y > u. \\
\end{cases}$

for an $(1 - \alpha)$PI of $[l, u]$ and the true value $y$ [@gneiting_strictly_2007, @bracher2021evaluating].

## Weighted Interval Score

The weighted interval score (WIS) is defined as

$\text{WIS}_{\alpha_{0:K}}(F, y) = \frac{1}{K+0.5}(w_0 \times |y - m| + \sum_{k=1}^K (w_k \times IS_{\alpha_k}(F, y)))$

where $m$ denotes the median prediction, $w_0$ denotes the weight of the median prediction, $IS_{\alpha_k}(F, y)$ denotes the interval score for the $1 - \alpha$ prediction interval and $w_k$ is the according weight. The WIS is calculated for a set of (central) PIs and the predictive median [@bracher2021evaluating]. The weights are an optional parameter and default weight is the canonical weight $w_k = \frac{2}{\alpha_k}$ and $w_0 = 0.5$. For these weights, it holds that:

$\text{WIS}_{\alpha_{0:K}}(F, y) \approx \text{CRPS}(F, y).$


::: scoringrules.interval_score

::: scoringrules.weighted_interval_score
