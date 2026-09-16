"""Load, represent and validate portfolio analysis configuration."""

from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from math import fsum, isclose, isfinite
from pathlib import Path
from typing import Any

import yaml


# Frozen dataclasses prevent accidental reassignment of configuration attributes
@dataclass(frozen=True)
class EstimationConfig:
    """Store historical estimation settings."""

    start_date: str
    end_date: str
    frequency: str
    periods_per_year: int


@dataclass(frozen=True)
class CashConfig:
    """Store assumptions for explicit portfolio cash."""

    expected_return: float


@dataclass(frozen=True)
class AssetConfig:
    """Store market-data metadata for one asset."""

    name: str
    ticker: str
    currency: str


@dataclass(frozen=True)
class PortfolioSnapshotConfig:
    """Store one named portfolio allocation."""

    name: str
    cash_weight: float
    weights: dict[str, float]


@dataclass(frozen=True)
class ProxyConfig:
    """Store configuration for a historical proxy series."""

    provider: str
    series_code: str
    apply_before_first_observation: bool


@dataclass(frozen=True)
class CorrelationStressConfig:
    """Store correlation-stress groups and scenario targets."""

    groups: dict[str, list[str]]
    target_correlations: tuple[float, ...]


@dataclass(frozen=True)
class PortfolioConfig:
    """Store the complete portfolio analysis configuration."""

    name: str
    base_currency: str
    estimation: EstimationConfig
    cash: CashConfig
    assets: dict[str, AssetConfig]
    portfolios: dict[str, PortfolioSnapshotConfig]
    proxies: dict[str, ProxyConfig]
    correlation_stress: CorrelationStressConfig | None


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    """Load and validate portfolio configuration from YAML."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        raw_config: Any = yaml.safe_load(file)

    # YAML is an external input boundary, so validate its expected root structure
    if not isinstance(raw_config, dict):
        raise ValueError("Configuration file must contain a top-level mapping.")

    portfolio = raw_config.get("portfolio")

    if not isinstance(portfolio, dict):
        raise ValueError("Configuration must contain a 'portfolio' mapping.")

    assets_data = _require_mapping(portfolio, "assets")
    portfolios_data = _require_mapping(portfolio, "portfolios")
    estimation_data = _require_mapping(portfolio, "estimation")
    cash_data = _require_mapping(portfolio, "cash")

    assets: dict[str, AssetConfig] = {}

    for asset_id, asset_data in assets_data.items():
        if not isinstance(asset_data, dict):
            raise ValueError(f"Asset '{asset_id}' configuration must be a mapping.")

        assets[asset_id] = AssetConfig(**asset_data)

    portfolios: dict[str, PortfolioSnapshotConfig] = {}

    for portfolio_id, snapshot_data in portfolios_data.items():
        if not isinstance(snapshot_data, dict):
            raise ValueError(
                f"Portfolio '{portfolio_id}' configuration must be a mapping."
            )

        weights_data = _require_mapping(snapshot_data, "weights")

        portfolios[portfolio_id] = PortfolioSnapshotConfig(
            name=snapshot_data["name"],
            cash_weight=float(snapshot_data["cash_weight"]),
            weights={
                asset_id: float(weight) for asset_id, weight in weights_data.items()
            },
        )

    proxies_data = portfolio.get("proxies", {})

    if not isinstance(proxies_data, dict):
        raise ValueError("'proxies' configuration must be a mapping.")

    proxies: dict[str, ProxyConfig] = {}

    for proxy_id, proxy_data in proxies_data.items():
        if not isinstance(proxy_data, dict):
            raise ValueError(f"Proxy '{proxy_id}' configuration must be a mapping.")

        proxies[proxy_id] = ProxyConfig(**proxy_data)

    stress_data = portfolio.get("correlation_stress")
    correlation_stress = None

    if stress_data is not None:
        if not isinstance(stress_data, dict):
            raise ValueError("'correlation_stress' configuration must be a mapping.")

        groups_data = _require_mapping(stress_data, "groups")

        correlation_stress = CorrelationStressConfig(
            groups={
                group_name: list(asset_ids)
                for group_name, asset_ids in groups_data.items()
            },
            target_correlations=tuple(
                float(value) for value in stress_data["target_correlations"]
            ),
        )

    config = PortfolioConfig(
        name=portfolio["name"],
        base_currency=portfolio["base_currency"],
        estimation=EstimationConfig(**estimation_data),
        cash=CashConfig(**cash_data),
        assets=assets,
        portfolios=portfolios,
        proxies=proxies,
        correlation_stress=correlation_stress,
    )

    _validate_portfolio_config(config)

    return config


def get_portfolio_snapshot(
    config: PortfolioConfig,
    portfolio_id: str,
) -> PortfolioSnapshotConfig:
    """Return a configured portfolio snapshot."""
    try:
        return config.portfolios[portfolio_id]
    except KeyError as error:
        available_snapshots = ", ".join(sorted(config.portfolios))
        raise ValueError(
            f"Unknown portfolio snapshot: '{portfolio_id}'. "
            f"Available snapshots: {available_snapshots}."
        ) from error


def get_portfolio_weights(
    config: PortfolioConfig,
    portfolio_id: str,
) -> dict[str, float]:
    """Return snapshot weights keyed by display asset name."""
    snapshot = get_portfolio_snapshot(
        config=config,
        portfolio_id=portfolio_id,
    )

    return {
        config.assets[asset_id].name: weight
        for asset_id, weight in snapshot.weights.items()
    }


def resolve_correlation_stress_groups(
    config: PortfolioConfig,
    portfolio_id: str,
) -> dict[str, list[str]]:
    """Resolve stress groups for assets held in one portfolio snapshot."""
    stress_config = config.correlation_stress

    if stress_config is None:
        raise ValueError("Correlation stress is not configured.")

    snapshot = get_portfolio_snapshot(
        config=config,
        portfolio_id=portfolio_id,
    )

    # Zero-weight entries have no exposure and should not activate stress groups
    active_asset_ids = {
        asset_id for asset_id, weight in snapshot.weights.items() if weight > 0.0
    }

    resolved_groups: dict[str, list[str]] = {}

    for group_name, asset_ids in stress_config.groups.items():
        active_members = [
            asset_id for asset_id in asset_ids if asset_id in active_asset_ids
        ]

        # Correlation requires at least one pair of active assets
        if len(active_members) < 2:
            continue

        resolved_groups[group_name] = [
            config.assets[asset_id].name for asset_id in active_members
        ]

    if not resolved_groups:
        raise ValueError(
            "No configured stress group contains at least two active portfolio assets."
        )

    return resolved_groups


def _require_mapping(
    data: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return a required YAML section after confirming it is a mapping."""
    value = data.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{key}' must be a mapping.")

    return value


def _validate_portfolio_config(config: PortfolioConfig) -> None:
    """Validate internal consistency of portfolio configuration."""
    if not config.name.strip():
        raise ValueError("Portfolio name cannot be empty.")

    if not config.base_currency.strip():
        raise ValueError("Base currency cannot be empty.")

    if not isfinite(config.cash.expected_return):
        raise ValueError("Cash expected return must be finite.")

    _validate_estimation_config(config.estimation)
    _validate_assets(config.assets)
    _validate_portfolio_snapshots(config)
    _validate_proxies(config)
    _validate_correlation_stress(config)


def _validate_estimation_config(config: EstimationConfig) -> None:
    """Validate historical estimation settings."""
    try:
        start_date = date.fromisoformat(config.start_date)
        end_date = date.fromisoformat(config.end_date)
    except (TypeError, ValueError) as error:
        raise ValueError("Estimation dates must use ISO format YYYY-MM-DD.") from error

    if start_date >= end_date:
        raise ValueError("Estimation start date must precede end date.")

    if not config.frequency.strip():
        raise ValueError("Estimation frequency cannot be empty.")

    if config.periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")


def _validate_assets(assets: dict[str, AssetConfig]) -> None:
    """Validate market-data asset definitions."""
    if not assets:
        raise ValueError("Portfolio must contain at least one asset.")

    for asset_id, asset in assets.items():
        if not asset.name.strip():
            raise ValueError(f"Asset '{asset_id}' must have a display name.")

        if not asset.ticker.strip():
            raise ValueError(f"Asset '{asset_id}' must have a ticker.")

        if not asset.currency.strip():
            raise ValueError(f"Asset '{asset_id}' must have a currency.")

    tickers = [asset.ticker for asset in assets.values()]

    if len(tickers) != len(set(tickers)):
        raise ValueError("Asset tickers must be unique.")

    display_names = [asset.name for asset in assets.values()]

    # Display names become data labels, so duplicates would create ambiguity
    if len(display_names) != len(set(display_names)):
        raise ValueError("Asset display names must be unique.")


def _validate_portfolio_snapshots(config: PortfolioConfig) -> None:
    """Validate configured portfolio allocations."""
    if not config.portfolios:
        raise ValueError("At least one portfolio snapshot must be configured.")

    asset_ids = set(config.assets)

    for portfolio_id, snapshot in config.portfolios.items():
        if not snapshot.name.strip():
            raise ValueError(f"Portfolio '{portfolio_id}' must have a name.")

        if not snapshot.weights:
            raise ValueError(f"Portfolio '{portfolio_id}' contains no risky assets.")

        unknown_assets = set(snapshot.weights) - asset_ids

        if unknown_assets:
            raise ValueError(
                f"Portfolio '{portfolio_id}' contains unknown assets: "
                f"{sorted(unknown_assets)}"
            )

        weights = tuple(snapshot.weights.values())

        if any(not isfinite(weight) for weight in weights):
            raise ValueError(f"Portfolio '{portfolio_id}' weights must be finite.")

        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"Portfolio '{portfolio_id}' weights cannot be negative.")

        if not any(weight > 0.0 for weight in weights):
            raise ValueError(
                f"Portfolio '{portfolio_id}' must contain a positive risky weight."
            )

        if not isfinite(snapshot.cash_weight) or snapshot.cash_weight < 0.0:
            raise ValueError(f"Portfolio '{portfolio_id}' cash weight is invalid.")

        # fsum reduces floating-point accumulation error across many positions
        total_weight = fsum((*weights, snapshot.cash_weight))

        if not isclose(
            total_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"Portfolio '{portfolio_id}' weights must sum to 1.0. "
                f"Received {total_weight:.6f}."
            )


def _validate_proxies(config: PortfolioConfig) -> None:
    """Validate historical proxy definitions."""
    unknown_assets = set(config.proxies) - set(config.assets)

    if unknown_assets:
        raise ValueError(
            f"Historical proxies reference unknown assets: {sorted(unknown_assets)}"
        )

    for proxy_id, proxy in config.proxies.items():
        if not proxy.provider.strip():
            raise ValueError(f"Proxy '{proxy_id}' must have a provider.")

        if not proxy.series_code.strip():
            raise ValueError(f"Proxy '{proxy_id}' must have a series code.")

        if not isinstance(proxy.apply_before_first_observation, bool):
            raise ValueError(
                f"Proxy '{proxy_id}' apply_before_first_observation must be boolean."
            )


def _validate_correlation_stress(config: PortfolioConfig) -> None:
    """Validate configured correlation-stress scenarios."""
    stress_config = config.correlation_stress

    if stress_config is None:
        return

    if not stress_config.groups:
        raise ValueError("Correlation stress must contain at least one group.")

    asset_ids = set(config.assets)

    for group_name, group_assets in stress_config.groups.items():
        if len(group_assets) < 2:
            raise ValueError(
                f"Correlation stress group '{group_name}' must contain "
                "at least two assets."
            )

        if len(group_assets) != len(set(group_assets)):
            raise ValueError(
                f"Correlation stress group '{group_name}' contains duplicate assets."
            )

        unknown_assets = set(group_assets) - asset_ids

        if unknown_assets:
            raise ValueError(
                f"Correlation stress group '{group_name}' contains unknown assets: "
                f"{sorted(unknown_assets)}"
            )

    targets = stress_config.target_correlations

    if not targets:
        raise ValueError("At least one correlation stress target is required.")

    if any(not isfinite(target) for target in targets):
        raise ValueError("Correlation stress targets must be finite.")

    if any(target < -1.0 or target >= 1.0 for target in targets):
        raise ValueError("Correlation stress targets must lie in the interval [-1, 1).")

    # pairwise compares each target with the value that immediately follows it
    if any(current >= following for current, following in pairwise(targets)):
        raise ValueError("Correlation stress targets must be strictly increasing.")
