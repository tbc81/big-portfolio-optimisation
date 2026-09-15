from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# Defines statistical sample window
@dataclass(frozen=True)
class EstimationConfig:
    start_date: str
    end_date: str
    frequency: str
    periods_per_year: int


# Tracks non-invested capital
@dataclass(frozen=True)
class CashConfig:
    current_weight: float
    expected_return: float


# Standardised profile for each investable risky asset
@dataclass(frozen=True)
class AssetConfig:
    name: str
    ticker: str
    currency: str
    current_weight: float


# Handles data for assets with insufficient historical records
@dataclass(frozen=True)
class ProxyConfig:
    provider: str
    series_code: str
    apply_before_first_observation: (
        bool  # flag determining whether to splice proxy returns
    )
    # into asset's pre-history


# Unifies portfolio-level metadata, estimation setup, cash profile and dictonaries
@dataclass(frozen=True)
class PortfolioConfig:
    name: str
    base_currency: str
    estimation: EstimationConfig
    cash: CashConfig
    assets: dict[str, AssetConfig]
    proxies: dict[str, ProxyConfig]


# Reads config file from disk, parses raw data into typed objects and validates it
def load_portfolio_config(
    path: str | Path,
) -> PortfolioConfig:
    """Load and validate portfolio configuration from YAML."""
    config_path = Path(path)  # Converts raw string path into structured Path object

    # Opens file in read-only mode, using UTF-8 character encoding
    with config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file)  # converts plaintext to
        # standard python dictionaries and lists

    portfolio = raw_config["portfolio"]

    # Construct typed asset configuration from YAML entries.
    assets = {
        asset_id: AssetConfig(**asset_data)
        for asset_id, asset_data in portfolio["assets"].items()
    }

    # Proxies are optional in the portfolio configuration
    proxies = {
        proxy_id: ProxyConfig(**proxy_data)
        for proxy_id, proxy_data in portfolio.get(
            "proxies",
            {},
        ).items()
    }

    config = PortfolioConfig(
        name=portfolio["name"],
        base_currency=portfolio["base_currency"],
        estimation=EstimationConfig(**portfolio["estimation"]),
        cash=CashConfig(**portfolio["cash"]),
        assets=assets,
        proxies=proxies,
    )

    _validate_portfolio_config(config)

    return config


def _validate_portfolio_config(
    config: PortfolioConfig,
) -> None:
    """Validate internal consistency of portfolio configuration."""
    asset_weight = sum(asset.current_weight for asset in config.assets.values())

    total_weight = asset_weight + config.cash.current_weight

    # Float weight verification
    if not abs(total_weight - 1.0) < 1e-10:
        raise ValueError(
            f"Portfolio weights must sum to 1.0. Received {total_weight:.6f}."
        )

    # Rejects empty portfolios ensuring optimisation calculations
    # do not construct zero dimensional matrices
    if len(config.assets) == 0:
        raise ValueError("Portfolio must contain at least one asset.")

    tickers = [asset.ticker for asset in config.assets.values()]

    # Checking ticker uniqueness and ensuring zero duplication
    if len(tickers) != len(set(tickers)):
        raise ValueError("Asset tickers must be unique.")

    # Guarantees annualisation parameters are strictly positive
    if config.estimation.periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
