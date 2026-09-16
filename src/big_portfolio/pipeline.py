"""Run the end-to-end portfolio analysis pipeline."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from big_portfolio.analysis.diagnostics import (
    PortfolioDiagnosticResult,
    analyse_portfolio_efficiency,
    build_allocation_comparison,
)
from big_portfolio.config import (
    PortfolioConfig,
    get_portfolio_snapshot,
    get_portfolio_weights,
    load_portfolio_config,
)
from big_portfolio.data.cash_proxy import (
    backfill_price_with_proxy,
    download_boe_series,
)
from big_portfolio.data.currency import convert_prices_to_gbp
from big_portfolio.data.market_data import (
    download_adjusted_prices,
    download_fx_rates,
)
from big_portfolio.data.returns import build_weekly_log_returns
from big_portfolio.estimation.moments import estimate_annual_moments
from big_portfolio.optimisation.mean_variance import (
    EfficientFrontierResult,
    generate_efficient_frontier,
)
from big_portfolio.visualisation.frontier import plot_efficient_frontier

BANK_OF_ENGLAND_PROVIDER = "Bank of England"

DEFAULT_RISK_AVERSIONS = tuple(
    float(value)
    for value in np.logspace(
        -1,
        1.5,
        40,
    )[::-1]
)


@dataclass(frozen=True)
class OutputPaths:
    """Store generated analysis output locations."""

    asset_statistics: Path
    covariance: Path
    allocation_comparison: Path
    frontier_summary: Path
    frontier_weights: Path
    efficient_frontier: Path


@dataclass(frozen=True)
class PipelineResult:
    """Store the main results from a complete portfolio analysis."""

    portfolio_id: str
    diagnostic: PortfolioDiagnosticResult
    frontier: EfficientFrontierResult
    asset_statistics: pd.DataFrame
    allocation_comparison: pd.DataFrame
    outputs: OutputPaths


def run_portfolio_analysis(
    config_path: str | Path = "config/portfolio.yaml",
    output_directory: str | Path = "outputs",
    risk_aversions: Iterable[float] | None = None,
    portfolio_id: str = "current",
) -> PipelineResult:
    """Run the configured portfolio analysis from market data to outputs."""
    config = load_portfolio_config(config_path)

    prices = download_adjusted_prices(
        assets=config.assets,
        start_date=config.estimation.start_date,
        end_date=config.estimation.end_date,
    )

    currencies = {asset.currency for asset in config.assets.values()}

    fx_rates = download_fx_rates(
        currencies=currencies,
        start_date=config.estimation.start_date,
        end_date=config.estimation.end_date,
    )

    gbp_prices = convert_prices_to_gbp(
        prices=prices,
        assets=config.assets,
        fx_rates=fx_rates,
    )

    analysis_prices = _apply_price_proxies(
        prices=gbp_prices,
        config=config,
    )

    # Complete weekly scenarios are used for joint return and covariance estimation
    return_result = build_weekly_log_returns(
        prices=analysis_prices,
        frequency=config.estimation.frequency,
    )

    estimate = estimate_annual_moments(
        log_returns=return_result.log_returns,
        periods_per_year=config.estimation.periods_per_year,
    )

    portfolio_snapshot = get_portfolio_snapshot(
        config=config,
        portfolio_id=portfolio_id,
    )

    portfolio_weights = pd.Series(
        get_portfolio_weights(
            config=config,
            portfolio_id=portfolio_id,
        ),
        name="portfolio_weight",
        dtype=float,
    )

    # Optimisation uses the asset universe held by the selected portfolio snapshot
    active_assets = portfolio_weights.index.tolist()

    active_expected_returns = estimate.expected_returns.loc[active_assets]
    active_covariance = estimate.covariance.loc[
        active_assets,
        active_assets,
    ]

    diagnostic = analyse_portfolio_efficiency(
        expected_returns=active_expected_returns,
        covariance=active_covariance,
        current_weights=portfolio_weights,
        current_cash_weight=portfolio_snapshot.cash_weight,
        cash_expected_return=config.cash.expected_return,
    )

    frontier_values = (
        DEFAULT_RISK_AVERSIONS if risk_aversions is None else risk_aversions
    )

    frontier = generate_efficient_frontier(
        expected_returns=active_expected_returns,
        covariance=active_covariance,
        risk_aversions=frontier_values,
        cash_expected_return=config.cash.expected_return,
    )

    asset_statistics = pd.DataFrame(
        {
            "annual_expected_return": active_expected_returns,
            "annual_volatility": estimate.volatilities.loc[active_assets],
        }
    )

    allocation_comparison = build_allocation_comparison(diagnostic)

    output_paths = _build_output_paths(
        output_directory=output_directory,
        portfolio_id=portfolio_id,
    )

    _save_tables(
        asset_statistics=asset_statistics,
        covariance=active_covariance,
        allocation_comparison=allocation_comparison,
        frontier=frontier,
        output_paths=output_paths,
    )

    figure = plot_efficient_frontier(
        frontier=frontier,
        diagnostic=diagnostic,
        output_path=output_paths.efficient_frontier,
    )
    plt.close(figure)

    return PipelineResult(
        portfolio_id=portfolio_id,
        diagnostic=diagnostic,
        frontier=frontier,
        asset_statistics=asset_statistics,
        allocation_comparison=allocation_comparison,
        outputs=output_paths,
    )


def _apply_price_proxies(
    prices: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.DataFrame:
    """Apply configured historical proxies to incomplete asset prices."""
    analysis_prices = prices

    for asset_id, proxy in config.proxies.items():
        if not proxy.apply_before_first_observation:
            continue

        if proxy.provider != BANK_OF_ENGLAND_PROVIDER:
            raise ValueError(
                f"Unsupported proxy provider for '{asset_id}': '{proxy.provider}'."
            )

        proxy_index = download_boe_series(
            series_code=proxy.series_code,
            start_date=config.estimation.start_date,
            end_date=config.estimation.end_date,
        )

        proxy_result = backfill_price_with_proxy(
            prices=analysis_prices,
            asset_name=config.assets[asset_id].name,
            proxy_index=proxy_index,
        )

        analysis_prices = proxy_result.prices

    return analysis_prices


def _build_output_paths(
    output_directory: str | Path,
    portfolio_id: str,
) -> OutputPaths:
    """Construct deterministic output paths for one portfolio analysis."""
    output_directory = Path(output_directory)

    table_directory = output_directory / "tables"
    figure_directory = output_directory / "figures"

    return OutputPaths(
        asset_statistics=(table_directory / f"asset_statistics_{portfolio_id}.csv"),
        covariance=(table_directory / f"covariance_{portfolio_id}.csv"),
        allocation_comparison=(
            table_directory / f"allocation_comparison_{portfolio_id}.csv"
        ),
        frontier_summary=(table_directory / f"frontier_summary_{portfolio_id}.csv"),
        frontier_weights=(table_directory / f"frontier_weights_{portfolio_id}.csv"),
        efficient_frontier=(
            figure_directory / f"efficient_frontier_{portfolio_id}.png"
        ),
    )


def _save_tables(
    asset_statistics: pd.DataFrame,
    covariance: pd.DataFrame,
    allocation_comparison: pd.DataFrame,
    frontier: EfficientFrontierResult,
    output_paths: OutputPaths,
) -> None:
    """Save the principal numerical outputs as CSV files."""
    output_paths.asset_statistics.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    asset_statistics.to_csv(output_paths.asset_statistics)
    covariance.to_csv(output_paths.covariance)
    allocation_comparison.to_csv(output_paths.allocation_comparison)
    frontier.summary.to_csv(
        output_paths.frontier_summary,
        index=False,
    )
    frontier.weights.to_csv(
        output_paths.frontier_weights,
        index=False,
    )
