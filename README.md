# BIG Portfolio Optimisation

An end-to-end portfolio optimisation and risk-analysis project developed around the real holdings of Bath Investors Group (BIG). The framework converts multi-currency market data into GBP, estimates annual return and covariance inputs from weekly observations, solves long-only mean-variance problems with MOSEK Fusion, and measures the sensitivity of portfolio risk to higher within-group correlations.

The repository is intended to show the full modelling process, including the assumptions that enter before the optimiser is called. Expected returns, covariance, currency treatment and stress construction all affect the final allocation, so each stage is kept explicit.

## What the project does

The analysis proceeds in six stages:

1. Download adjusted market prices and the required FX series.
2. Convert all risky assets into GBP.
3. Apply configured historical proxies where an asset has insufficient observed history.
4. Construct complete weekly log-return scenarios.
5. Estimate annual arithmetic expected returns and covariance.
6. Evaluate the observed portfolio, solve mean-variance alternatives, and apply correlation stresses.

The current configuration contains separate `current` and `pre_diversification` portfolio snapshots, allowing the same framework to compare the two allocations.

## Documentation

The full methodology, mathematical derivations, case-study results, model limitations and planned analytical extensions are documented separately.

- [Data preparation](docs/methodology/data.md)
- [Moment estimation](docs/methodology/estimation.md)
- [Mean-variance optimisation](docs/methodology/optimisation.md)
- [Correlation stress analysis](docs/methodology/correlation-stress.md)
- [BIG case study](docs/case-study/big-portfolio.md)
- [Limitations and analytical roadmap](docs/development/roadmap.md)

Full technical documentation is available on the [project website](https://tbc81.github.io/big-portfolio-optimisation/).

## Installation

The project requires Python 3.12+ and a valid MOSEK licence.

```bash
git clone https://github.com/tbc81/big-portfolio-optimisation.git
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

Generated tables and figures are written to `outputs/` by default. This directory is excluded from version control and can be replaced with any destination through `--output-dir`.

Portfolio definitions, currencies, proxies and risk groups are stored in:

```text
config/portfolio.yaml
```

## Project structure

```text
big-portfolio-optimisation/
├── config/
│   └── portfolio.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── assets/
│   │   └── figures/
│   ├── case-study/
│   │   └── big-portfolio.md
│   ├── development/
│   │   └── roadmap.md
│   ├── javascripts/
│   │   └── mathjax.js
│   ├── methodology/
│   │   ├── correlation-stress.md
│   │   ├── data.md
│   │   ├── estimation.md
│   │   └── optimisation.md
│   └── index.md
├── src/
│   └── big_portfolio/
│       ├── analysis/
│       ├── data/
│       ├── estimation/
│       ├── optimisation/
│       ├── visualisation/
│       ├── cli.py
│       ├── config.py
│       └── pipeline.py
├── tests/
├── .gitignore
├── LICENSE
├── mkdocs.yml
├── pyproject.toml
└── README.md
```

The package follows a `src/` layout so importable project code is separated from configuration, tests, documentation and generated outputs.

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

## Licence

This project is released under the MIT Licence.

## Disclaimer

This repository is an educational and analytical project based on a student-run portfolio. Historical estimates, efficient allocations and stress results depend on the data sample and modelling assumptions and do not constitute investment advice.
