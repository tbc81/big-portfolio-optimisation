# BIG Portfolio Optimisation

An end-to-end portfolio optimisation and risk-analysis project developed around the real holdings of Bath Investors Group (BIG). The framework converts multi-currency market data into GBP, estimates annual return and covariance inputs from weekly observations, solves long-only mean-variance problems with the MOSEK Fusion API, and measures the sensitivity of portfolio risk to higher within-group correlations.

The repository is intended to show the full modelling process, including the assumptions that enter before the optimiser is called. Expected returns, covariance, currency treatment and stress construction all affect the final allocation, so each stage is kept explicit.

![Efficient frontier for the current BIG portfolio](outputs/figures/efficient_frontier_current.png)

## Contents

- [Methodology](#methodology)
- [Data preparation](#data-preparation)
  - [Adjusted prices](#adjusted-prices)
  - [Currency conversion](#currency-conversion)
  - [Historical proxy for CSH2](#historical-proxy-for-csh2)
  - [Weekly sampling](#weekly-sampling)
  - [Complete return scenarios](#complete-return-scenarios)
- [Moment estimation](#moment-estimation)
  - [Conversion to arithmetic moments](#conversion-to-arithmetic-moments)
- [Portfolio model](#portfolio-model)
  - [Expected return](#expected-return)
  - [Portfolio risk](#portfolio-risk)
- [Mean-variance optimisation](#mean-variance-optimisation)
  - [Conic representation](#conic-representation)
  - [Maximum return subject to a volatility limit](#maximum-return-subject-to-a-volatility-limit)
  - [Minimum volatility subject to a return floor](#minimum-volatility-subject-to-a-return-floor)
  - [Efficient frontier](#efficient-frontier)
  - [Why MOSEK Fusion is used](#why-mosek-fusion-is-used)
- [Correlation stress analysis](#correlation-stress-analysis)
  - [From covariance to correlation](#from-covariance-to-correlation)
  - [Correlation floors](#correlation-floors)
  - [Why the full matrix must be solved](#why-the-full-matrix-must-be-solved)
  - [Preserve marginal volatility](#preserve-marginal-volatility)
  - [Portfolio sensitivity](#portfolio-sensitivity)
- [Case-study results](#case-study-results)
- [Limitations](#limitations)
- [Planned analytical extensions](#planned-analytical-extensions)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Generated outputs](#generated-outputs)
- [Testing and code quality](#testing-and-code-quality)
- [Disclaimer](#disclaimer)

## Methodology

The analysis proceeds in six stages:

1. Download adjusted market prices and the required FX series.
2. Convert all risky assets into GBP.
3. Apply configured historical proxies where an asset has insufficient observed history.
4. Construct complete weekly log-return scenarios.
5. Estimate annual arithmetic expected returns and covariance.
6. Evaluate the observed portfolio, solve mean-variance alternatives, and apply correlation stresses.

Portfolio definitions, currencies, proxies and risk groups are stored in `config/portfolio.yaml`. The current configuration contains separate `current` and `pre_diversification` portfolio snapshots, allowing the same model to compare the two allocations.

## Data preparation

### Adjusted prices

Adjusted prices are downloaded with `yfinance`. Corporate actions such as stock splits and distributions can otherwise introduce price changes unrelated to investment performance.

The configured universe contains assets quoted in GBP, GBX, EUR, USD and CAD. All prices are converted into GBP before returns are calculated.

### Currency conversion

For a UK investor, the return on a foreign asset depends on both the local asset price and the exchange rate. Converting the price series first ensures that both effects enter the GBP return.

The current implementation applies the quote convention of each downloaded FX series:

- GBX prices are divided by 100 to convert pence into pounds;
- EUR prices are multiplied by EUR/GBP;
- USD prices are divided by GBP/USD;
- CAD prices are divided by GBP/CAD.

For example, a US equity can rise in USD whilst falling in GBP terms if sterling appreciates sufficiently over the same period. The model therefore estimates risk and return from the investor's GBP price path.

### Historical proxy for CSH2

CSH2 has a shorter observed history than the main estimation sample. Its pre-history is extended using the Bank of England SONIA Compounded Index, as specified in the configuration.

Let $P_{t_0}$ denote the first observed CSH2 market price and let $I_{t_0}$ denote the corresponding SONIA index value. The scaling factor is

$$
s = \frac{P_{t_0}}{I_{t_0}}.
$$

For dates before $t_0$, the proxy price is

$$
\widetilde{P}_t = s I_t.
$$

The scaling makes the proxy continuous with the first observed market price. Observed CSH2 prices are left unchanged. For unmatched dates, the implementation uses the latest proxy observation available on or before the asset date, avoiding look-ahead bias.

This treatment supplies a cash-like historical series for the missing period. It is an approximation of the pre-history; realised CSH2 trading behaviour also reflects fees, tracking effects and market pricing.

### Weekly sampling

Daily prices are resampled to `W-FRI` using the last available market observation in each weekly period.

Weekly sampling retains substantially more observations than monthly data whilst reducing some daily noise and mismatches in calendars across markets. The choice also keeps the covariance estimate based on a reasonably large joint sample. The frequency and annualisation factor remain configurable.

### Complete return scenarios

For weekly GBP price $P_t$, the log return is

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right).
$$

Log returns are additive through time. If a price moves from $P_0$ to $P_1$ and then from $P_1$ to $P_2$,

$$
\ln\left(\frac{P_1}{P_0}\right)
+
\ln\left(\frac{P_2}{P_1}\right)
=
\ln\left(\frac{P_2}{P_0}\right).
$$

Covariance requires returns for all assets in the same observation period. Weeks containing a missing return for any asset are therefore removed before moment estimation. This gives a complete joint scenario matrix. The cost is a smaller effective sample when one asset has sparse history.

## Moment estimation

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

### Conversion to arithmetic moments

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

These expected returns are historical sample estimates implied by the lognormal model. Forecasting future returns lies outside the current estimation framework.

## Portfolio model

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

### Expected return

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

### Portfolio risk

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

The diagonal entries of $\Sigma$ are individual asset variances. The off-diagonal entries are covariances and determine how strongly assets move together. Portfolio diversification therefore depends on both individual volatility and cross-asset co-movement.

## Mean-variance optimisation

### Conic representation

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

### Maximum return subject to a volatility limit

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

### Minimum volatility subject to a return floor

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

### Efficient frontier

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

### Why MOSEK Fusion is used

The two main optimisation tasks in the project have standard convex formulations. Portfolio volatility is represented with a second-order cone, and the correlation-stress problem uses a positive-semidefinite cone.

Fusion exposes these mathematical objects directly through variables, expressions, domains and constraints. The solver returns the optimal solution for the stated convex problem subject to numerical tolerances. Portfolio volatility is then recalculated from

$$
\sqrt{x^T\Sigma x}
$$

using the returned weights.

## Correlation stress analysis

Historical covariance can understate risk if assets that previously diversified one another become more correlated. The stress module isolates this channel by increasing selected correlations while holding individual asset volatilities fixed.

The current BIG case study defines a technology/growth risk group in configuration. Stress groups may overlap; each selected asset pair is constrained once.

### From covariance to correlation

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

### Correlation floors

For every selected pair $(i,j)$, a stress scenario imposes

$$
C^*_{ij}
\geq
\rho_{\text{target}}.
$$

A target of $0.85$, for example, requires each selected pair to have correlation of at least 85% in the stressed matrix.

The targets currently used in the case study are sensitivity parameters. Empirical calibration to historical rolling-correlation quantiles is the next planned extension.

### Why the full matrix must be solved

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

It measures the total entry-by-entry change in the matrix. Minimising this distance limits changes outside the stressed pairs, to those required by the correlation floors and matrix-validity constraints.

The condition

$$
C^*-\varepsilon I \succeq 0
$$

sets a small positive lower bound on the eigenvalues. This keeps the stressed matrix suitable for the Cholesky factorisation required by the portfolio optimiser.

### Preserve marginal volatility

The stressed covariance matrix is reconstructed as

$$
\Sigma^*
=
DC^*D.
$$

The same volatility matrix $D$ is used before and after the stress. The scenario therefore changes correlation while preserving each asset's marginal volatility.

This is a deliberate isolation of one source of risk. A broader market stress could also change asset volatility, expected return, FX rates or liquidity.

### Portfolio sensitivity

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

## Case-study results

The current and pre-diversification BIG snapshots have different exposure to the configured technology/growth group:

| Portfolio snapshot | Technology/growth exposure |
| --- | ---: |
| Current | 27.5% |
| Pre-diversification | 35.0% |

Under an 85% within-group correlation floor, the verified results are:

| Portfolio snapshot | Base volatility | Stressed volatility | Variance increase |
| --- | ---: | ---: | ---: |
| Current | 22.94% | 25.70% | 25.6% |
| Pre-diversification | 26.94% | 31.25% | 34.6% |

The stress-induced volatility increases are approximately

$$
25.70\%-22.94\%=2.76
\text{ percentage points}
$$

for the current portfolio and

$$
31.25\%-26.94\%=4.31
\text{ percentage points}
$$

for the pre-diversification portfolio.

The reduction in the stress-induced increase is therefore approximately

$$
1-\frac{2.76}{4.31}
\approx 36\%.
$$

Under this specified correlation stress, the current portfolio has both lower base volatility and a smaller increase in volatility. The calculation is a conditional sensitivity measure based on the estimated covariance structure and the imposed 85% floor; future realised volatility may differ.

## Limitations

### Expected-return estimation

Historical expected returns are the least stable input in the current model. Small changes in $\mu$ can cause large changes in optimal weights because the optimiser responds directly to cross-sectional differences in estimated return.

This is particularly important for the maximum-return and frontier solutions. Their allocations should be interpreted as model-implied outputs under the estimated sample moments.

### Covariance estimation

The project currently uses the sample covariance implied by the historical return scenarios. With a finite sample, individual covariance estimates contain noise and can produce unstable optimiser weights.

Positive-definiteness checks establish that the matrix is numerically usable. Estimation precision remains a separate statistical issue.

### Distributional and time-series assumptions

The lognormal moment conversion and linear annualisation impose a stable return model over the estimation period. Financial returns can display fat tails, volatility clustering and structural changes in correlation.

These time-varying features are outside the current model.

### Implementation constraints

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

### Stress calibration

The current correlation floors are specified sensitivity levels. The analysis quantifies the effect of a given correlation regime; its historical likelihood remains to be estimated.

## Planned analytical extensions

The next stage focuses on estimation quality and model validation.

### 1. Empirical correlation calibration

The next priority is to calculate rolling within-group correlations from the existing weekly GBP return scenarios.

The analysis will compare alternative windows, initially 13, 26 and 52 weeks, and examine upper quantiles such as the 75th, 90th and 95th percentiles. This will allow common stress floors to be linked to historical correlation regimes.

The choice of window requires sensitivity analysis. Short windows adapt quickly but are noisy; long windows are smoother but respond slowly to regime changes.

### 2. Covariance shrinkage

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

### 3. Expected-return estimation

Possible extensions include shrinkage of historical means, Bayesian estimates, Black-Litterman style views and explicit investment-committee assumptions.

Because the mean vector has a large effect on optimised allocations, each method should be compared through stability and out-of-sample performance.

### 4. Walk-forward validation

A walk-forward exercise would repeatedly estimate the model using information available up to time $t$, solve the portfolio, and evaluate the resulting allocation over a later holdout period.

This provides a direct test of whether an estimation or optimisation change improves performance outside the sample used to fit it.

### 5. Time-varying risk estimates

Potential extensions include exponentially weighted covariance, rolling covariance and regime-conditioned covariance matrices.

These approaches allow recent observations or identified regimes to influence the risk model more strongly. Their value should be assessed against out-of-sample stability.

### 6. Portfolio implementation constraints

Maximum position sizes, risk-group limits, turnover constraints, transaction costs and benchmark-relative limits would bring the optimisation closer to an implementable portfolio decision.

These constraints also provide a useful test of whether the current efficient solutions depend on concentrated or high-turnover allocations.

### 7. Robust optimisation and factor models

Robust optimisation can incorporate uncertainty in estimated returns or covariance directly into the optimisation problem. A factor model can estimate risk through a smaller set of common drivers and may improve covariance stability.

Both extensions require calibration and should be judged on whether they improve out-of-sample behaviour.

### 8. Broader stress scenarios

The current stress engine isolates correlation. Later scenarios can add shocks to marginal volatility, expected returns, FX rates or liquidity while keeping each component separately identifiable.

## Project structure

```text
big-portfolio-optimisation/
├── config/
│   └── portfolio.yaml
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   └── big_portfolio/
│       ├── analysis/
│       │   ├── correlation_stress.py
│       │   └── diagnostics.py
│       ├── data/
│       │   ├── cash_proxy.py
│       │   ├── currency.py
│       │   ├── market_data.py
│       │   └── returns.py
│       ├── estimation/
│       │   └── moments.py
│       ├── optimisation/
│       │   └── mean_variance.py
│       ├── visualisation/
│       │   ├── correlation_stress_visualisation.py
│       │   └── frontier.py
│       ├── cli.py
│       ├── config.py
│       └── pipeline.py
├── tests/
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Installation

The project requires Python 3.12+ and a valid MOSEK licence.

```bash
git clone <repository-url>
cd big-portfolio-optimisation

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -e ".[dev]"
```

The MOSEK licence must remain outside version control.

## Usage

Run the default `current` portfolio analysis:

```bash
portfolio-analytics run
```

Run the pre-diversification snapshot:

```bash
portfolio-analytics run --portfolio-id pre_diversification
```

Use a custom configuration or output directory:

```bash
portfolio-analytics run \
    --config path/to/portfolio.yaml \
    --output-dir results \
    --portfolio-id current
```

The CLI currently runs the market-data, estimation, portfolio-diagnostic and efficient-frontier pipeline. Correlation stress is implemented as a separate analysis module and can be run programmatically for multiple portfolio snapshots.

## Generated outputs

For the `current` snapshot, the pipeline writes:

```text
outputs/
├── figures/
│   └── efficient_frontier_current.png
└── tables/
    ├── allocation_comparison_current.csv
    ├── asset_statistics_current.csv
    ├── covariance_current.csv
    ├── frontier_summary_current.csv
    └── frontier_weights_current.csv
```

The portfolio identifier is included in each filename so that outputs from different configured snapshots can coexist.

## Testing and code quality

Run the full test suite:

```bash
pytest
```

Run MOSEK-dependent tests:

```bash
pytest -m mosek
```

Run Ruff:

```bash
ruff check src tests
ruff format src tests
```

The tests cover configuration, market data, currency conversion, proxy backfilling, return construction, moment estimation, optimisation constraints, correlation stress, diagnostics, visualisation, pipeline orchestration and CLI behaviour.

## Disclaimer

This repository is an educational and analytical project based on a student-run portfolio. Historical estimates, efficient allocations and stress results depend on the data sample and modelling assumptions and do not constitute investment advice.