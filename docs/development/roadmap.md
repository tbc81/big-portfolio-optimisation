# Limitations

## Expected-return estimation

Historical expected returns are the least stable input in the current model. Small changes in $\mu$ can cause large changes in optimal weights because the optimiser responds directly to cross-sectional differences in estimated return.

This is particularly important for the maximum-return and frontier solutions. Their allocations should be interpreted as model-implied outputs under the estimated sample moments.

## Covariance estimation

The project currently uses the sample covariance implied by the historical return scenarios. With a finite sample, individual covariance estimates contain noise and can produce unstable optimiser weights.

Positive-definiteness checks establish that the matrix is numerically usable. They do not establish that the matrix is estimated precisely.

## Distributional and time-series assumptions

The lognormal moment conversion and linear annualisation impose a stable return model over the estimation period. Financial returns can display fat tails, volatility clustering and structural changes in correlation.

The current framework does not estimate these features explicitly.

## Implementation constraints

The optimisation currently assumes:

- long-only positions;
- full investment;
- zero transaction costs;
- no turnover constraint;
- no position-size limits;
- no benchmark-relative constraint;
- no factor constraint;
- zero covariance for cash.

These assumptions keep the optimisation model transparent. They also allow allocations that may be costly or impractical to implement in a live portfolio.

## Stress calibration

The current correlation floors are specified sensitivity levels. The analysis establishes the effect of a given correlation regime; it does not estimate the probability of entering that regime.

# Planned analytical extensions

The next stage focuses on estimation quality and model validation.

## 1. Empirical correlation calibration

The next priority is to calculate rolling within-group correlations from the existing weekly GBP return scenarios.

The analysis will compare alternative windows, initially 13, 26 and 52 weeks, and examine upper quantiles such as the 75th, 90th and 95th percentiles. This will allow common stress floors to be linked to historical correlation regimes.

The choice of window requires sensitivity analysis. Short windows adapt quickly but are noisy; long windows are smoother but respond slowly to regime changes.

## 2. Covariance shrinkage

A shrinkage estimator can combine the sample covariance with a structured target:

$$
\Sigma_{\text{shrunk}}
=
(1-\lambda)\Sigma_{\text{sample}}
+
\lambda\Sigma_{\text{target}},
$$

where

$$
0\leq\lambda\leq1.
$$

Ledoit-Wolf shrinkage is a natural candidate. The objective is to improve conditioning and reduce the sensitivity of optimised weights to sampling noise.

## 3. Expected-return estimation

Possible extensions include shrinkage of historical means, Bayesian estimates, Black-Litterman style views and explicit investment-committee assumptions.

Because the mean vector has a large effect on optimised allocations, each method should be compared through stability and out-of-sample performance.

## 4. Walk-forward validation

A walk-forward exercise would repeatedly estimate the model using information available up to time $t$, solve the portfolio, and evaluate the resulting allocation over a later holdout period.

This provides a direct test of whether an estimation or optimisation change improves performance outside the sample used to fit it.

## 5. Time-varying risk estimates

Potential extensions include exponentially weighted covariance, rolling covariance and regime-conditioned covariance matrices.

These approaches allow recent observations or identified regimes to influence the risk model more strongly. Their value should be assessed against out-of-sample stability.

## 6. Portfolio implementation constraints

Maximum position sizes, risk-group limits, turnover constraints, transaction costs and benchmark-relative limits would bring the optimisation closer to an implementable portfolio decision.

These constraints also provide a useful test of whether the current efficient solutions depend on concentrated or high-turnover allocations.

## 7. Robust optimisation and factor models

Robust optimisation can incorporate uncertainty in estimated returns or covariance directly into the optimisation problem. A factor model can estimate risk through a smaller set of common drivers and may improve covariance stability.

Both extensions require calibration and should be judged on whether they improve out-of-sample behaviour.

## 8. Broader stress scenarios

The current stress engine isolates correlation. Later scenarios can add shocks to marginal volatility, expected returns, FX rates or liquidity while keeping each component separately identifiable.