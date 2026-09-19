# Moment estimation

Let $\mu_{\log,w}$ and $\Sigma_{\log,w}$ denote the sample mean vector and covariance matrix of weekly log returns. With $m$ periods per year,

$$
\mu_{\log,a} = m\mu_{\log,w}
$$

and

$$
\Sigma_{\log,a} = m\Sigma_{\log,w}.
$$

The current weekly configuration uses $m=52$.

This annualisation assumes that the weekly return-generating process is sufficiently stable for means and covariances to scale linearly through time. Time-varying volatility and correlation can weaken this approximation, which motivates several of the extensions discussed below.

## Conversion to arithmetic moments

The optimiser is expressed in arithmetic returns. Under a joint lognormal model, annual log moments can be converted analytically.

For asset $i$,

$$
\mathbb{E}[1+R_i]
=
\exp\left(
\mu_i + \frac{1}{2}\Sigma_{ii}
\right),
$$

hence

$$
\mathbb{E}[R_i]
=
\exp\left(
\mu_i + \frac{1}{2}\Sigma_{ii}
\right)-1.
$$

For assets $i$ and $j$,

$$
\operatorname{Cov}(R_i,R_j)
=
\mathbb{E}[1+R_i]\mathbb{E}[1+R_j]
\left(
\exp(\Sigma_{ij})-1
\right).
$$

The resulting arithmetic covariance matrix is checked for finite values, symmetry and positive definiteness before it is passed to the optimiser.

These expected returns are historical sample estimates implied by the lognormal model. They are model inputs, not forecasts.