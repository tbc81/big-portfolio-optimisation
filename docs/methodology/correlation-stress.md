# Correlation stress analysis

Historical covariance can understate risk if assets that previously diversified one another become more correlated. The stress module isolates this channel by increasing selected correlations while holding individual asset volatilities fixed.

The current BIG case study defines a technology/growth risk group in configuration. Stress groups may overlap; each selected asset pair is constrained once.

## From covariance to correlation

For assets $i$ and $j$,

$$
C_{ij}
=
\frac{\Sigma_{ij}}{\sigma_i\sigma_j}.
$$

Let

$$
D
=
\operatorname{diag}
(\sigma_1,\ldots,\sigma_n).
$$

Then

$$
\Sigma = DCD.
$$

Correlation separates co-movement from the scale of individual asset volatility.

## Correlation floors

For every selected pair $(i,j)$, a stress scenario imposes

$$
C^*_{ij}
\geq
\rho_{\text{target}}.
$$

A target of $0.85$, for example, requires each selected pair to have correlation of at least 85% in the stressed matrix.

The targets currently used in the case study are sensitivity parameters. They have not yet been calibrated to historical rolling-correlation quantiles.

## Why the full matrix must be solved

A correlation matrix must be jointly valid. Changing selected pairwise entries directly can produce a matrix with negative eigenvalues, even when each individual entry lies between $-1$ and $1$.

The model therefore solves for the nearest valid stressed matrix. Let $C$ denote the base correlation matrix and $C^*$ the stressed matrix. MOSEK solves

$$
\begin{aligned}
\min_{C^*}\quad
& \left\|C^*-C\right\|_F \\
\text{subject to}\quad
& \operatorname{diag}(C^*)=\mathbf{1} \\
& C^*-\varepsilon I \succeq 0 \\
& C^*_{ij}\geq\rho_{\text{target}}
\quad \text{for selected pairs},
\end{aligned}
$$

where

$$
\varepsilon = 10^{-6}.
$$

The Frobenius norm is

$$
\|A\|_F
=
\sqrt{
\sum_i\sum_j A_{ij}^2
}.
$$

It measures the total entry-by-entry change in the matrix. Minimising this distance limits changes outside the stressed pairs to those required by the correlation floors and matrix-validity constraints.

The condition

$$
C^*-\varepsilon I \succeq 0
$$

sets a small positive lower bound on the eigenvalues. This keeps the stressed matrix suitable for the Cholesky factorisation required by the portfolio optimiser.

## Preserve marginal volatility

The stressed covariance matrix is reconstructed as

$$
\Sigma^*
=
DC^*D.
$$

The same volatility matrix $D$ is used before and after the stress. The scenario therefore changes correlation while preserving each asset's marginal volatility.

This is a deliberate isolation of one source of risk. A broader market stress could also change asset volatility, expected return, FX rates or liquidity.

## Portfolio sensitivity

The observed portfolio weights are held fixed and volatility is recalculated under $\Sigma^*$.

The variance multiplier is

$$
M_{\text{var}}
=
\frac{x^T\Sigma^*x}
{x^T\Sigma x}
=
\left(
\frac{\sigma_{p,\text{stress}}}
{\sigma_{p,\text{base}}}
\right)^2.
$$

A multiplier of $1.25$ corresponds to a 25% increase in modelled portfolio variance.

For each stress scenario, the minimum-risk optimisation is also solved again at the original portfolio expected-return target. This separates two questions:

1. how much risk the held portfolio acquires under the stressed correlation structure;
2. how the same-return efficient allocation changes under that risk model.