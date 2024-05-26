# Interval Score

For a prediction interval (PI), the interval or Winkler score is given by:

$\text{IS} = \begin{cases}
    (u - l) + 2\frac{2}{\alpha}(l - y)  & \text{for } y < l \\
    (u - l)                             & \text{for } l \leq y \leq u \\
    (u - l) + \frac{2}{\alpha}(y - u)   & \text{for } y > u. \\
\end{cases}$

for an $(1 - \alpha)$PI of $[l, u]$ and the true value $y$ [@gneiting_strictly_2007, @bracher2021evaluating].

::: scoringrules.interval_score
