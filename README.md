# BIG Portfolio Optimisation

A multi-asset portfolio optimisation and risk-analysis framework built in Python
using the MOSEK Fusion API and applied to the Bath Investors Group portfolio.

## Overview

The project constructs a reproducible portfolio-optimisation pipeline from
historical market data through to mean-variance optimisation and portfolio
diagnostics.

The current model:

- downloads adjusted market prices for the portfolio holdings;
- converts multi-currency assets into GBP;
- constructs a SONIA-based historical proxy for CSH2;
- estimates annual expected returns and covariance from weekly log returns;
- generates a long-only mean-variance efficient frontier using MOSEK Fusion;
- compares the current portfolio with efficient portfolios at equivalent
  expected return and volatility.

## Current portfolio

The investment universe contains 18 securities across UK, European, US and
Canadian markets, together with a 1.5% cash allocation.

## Project structure

```text
config/             Portfolio and model configuration
src/big_portfolio/  Application source code
tests/              Automated tests
data/               Generated market and processed data
outputs/            Generated figures and analytical outputs
legacy/             Validated pre-refactor implementation
```
## Methodology
The optimisation framework follows mean-variance portfolio theory and uses
MOSEK Fusion to solve the resulting conic optimisation problems.

All foreign currency securities are converted into GBP before return 
estimation, allowing currency movements to enter the estimated portfolio
risk.

Historical estimates are model inputs and should not be interpreted as
forecasts of future investment performance. 

## Status
The core modelling pipeline has been validated. The project is currently being
refactored into a reusable Python package with automated testing and a command
line interface.