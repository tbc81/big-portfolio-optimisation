# Portfolio model

Let

- $x$ be the vector of risky-asset weights;
- $c$ be the cash weight;
- $\mu$ be the annual arithmetic expected-return vector;
- $r_c$ be the expected return assigned to cash;
- $\Sigma$ be the annual arithmetic covariance matrix.

The budget constraint is

$$
\mathbf{1}^T x + c = 1,
$$

with

$$
x \geq 0,
\qquad
c \geq 0.
$$

The model is therefore fully invested and long-only. Cash is represented separately with zero covariance against the risky assets. Its expected return is currently set to zero in the configuration.

## Expected return

Portfolio expected return is

$$
\mathbb{E}[R_p]
=
\mu^T x + r_c c.
$$

For a reader unfamiliar with vector notation,

$$
\mu^T x
=
\mu_1x_1+\mu_2x_2+\cdots+\mu_nx_n.
$$

Each asset's expected return is multiplied by its portfolio weight and the contributions are summed.

## Portfolio risk

Portfolio variance is

$$
\sigma_p^2
=
x^T\Sigma x,
$$

so annual volatility is

$$
\sigma_p
=
\sqrt{x^T\Sigma x}.
$$

The diagonal entries of $\Sigma$ are individual asset variances. The off-diagonal entries are covariances and determine how strongly assets move together. Diversification therefore depends on the full covariance matrix, not only on the volatility of each holding.

# Mean-variance optimisation

## Conic representation

The covariance matrix is factorised by Cholesky decomposition:

$$
\Sigma = GG^T.
$$

Substituting into the variance expression gives

$$
x^T\Sigma x
=
x^TGG^Tx
=
\left\|G^Tx\right\|_2^2.
$$

Therefore,

$$
\sigma_p
=
\left\|G^Tx\right\|_2.
$$

For a vector $z$, the Euclidean norm is

$$
\|z\|_2
=
\sqrt{z_1^2+z_2^2+\cdots+z_n^2}.
$$

This converts the portfolio risk constraint into a second-order cone, which MOSEK can solve directly.

## Maximum return subject to a volatility limit

The first diagnostic asks how much expected return the estimated model permits when the observed portfolio volatility is used as an upper bound.

The problem is

$$
\begin{aligned}
\max_{x,c}\quad
& \mu^T x + r_c c \\
\text{subject to}\quad
& \left\|G^Tx\right\|_2 \leq \gamma \\
& \mathbf{1}^T x + c = 1 \\
& x \geq 0 \\
& c \geq 0,
\end{aligned}
$$

where $\gamma$ is the observed portfolio volatility.

The reported return improvement is

$$
\Delta \mu
=
\mathbb{E}[R_{\text{same risk}}]
-
\mathbb{E}[R_{\text{portfolio}}].
$$

This comparison is conditional on the estimated $\mu$ and $\Sigma$. Since historical mean returns are noisy, the magnitude should be interpreted as a model diagnostic.

## Minimum volatility subject to a return floor

The second diagnostic fixes the observed portfolio expected return as a lower bound and minimises volatility.

Introducing an auxiliary risk variable $t$,

$$
\begin{aligned}
\min_{x,c,t}\quad
& t \\
\text{subject to}\quad
& \mu^T x + r_c c \geq R_{\text{target}} \\
& \left\|G^Tx\right\|_2 \leq t \\
& \mathbf{1}^T x + c = 1 \\
& x \geq 0 \\
& c \geq 0.
\end{aligned}
$$

The target is the expected return of the observed portfolio. The reported volatility reduction is

$$
\Delta \sigma
=
\sigma_{\text{portfolio}}
-
\sigma_{\text{same return}}.
$$

This comparison depends more directly on covariance than on the fixed-risk return diagnostic, although the target return still depends on the estimated mean vector.

## Efficient frontier

The efficient frontier is generated from

$$
\max_{x,c,t}
\quad
\mu^T x + r_c c - \alpha t
$$

subject to

$$
\left\|G^Tx\right\|_2 \leq t
$$

and the same budget and long-only constraints.

The parameter $\alpha \geq 0$ controls the penalty on volatility. Lower values place more weight on expected return; higher values place more weight on risk. The pipeline solves a logarithmically spaced grid of $\alpha$ values and reuses the same MOSEK model through a Fusion `Parameter`.

The risk penalty is expressed in volatility units, matching the risk measure reported throughout the project. A variance-penalty formulation is also common in mean-variance analysis; its risk-aversion parameter has a different scale and interpretation.

## Why MOSEK Fusion is used

The two main optimisation tasks in the project have standard convex formulations. Portfolio volatility is represented with a second-order cone, and the correlation-stress problem uses a positive-semidefinite cone.

Fusion exposes these mathematical objects directly through variables, expressions, domains and constraints. The solver returns the optimal solution for the stated convex problem subject to numerical tolerances. Portfolio volatility is then recalculated from

$$
\sqrt{x^T\Sigma x}
$$

using the returned weights.