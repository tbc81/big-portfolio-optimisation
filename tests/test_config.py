"""Test configuration loading, validation and snapshot resolution."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from big_portfolio.config import (
    get_portfolio_snapshot,
    get_portfolio_weights,
    load_portfolio_config,
    resolve_correlation_stress_groups,
)

CONFIG_PATH = Path("config/portfolio.yaml")


def _load_raw_config() -> dict[str, Any]:
    """Load the project YAML as mutable test data."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise TypeError("Test configuration must contain a top-level mapping.")

    return raw_config


def _write_test_config(
    tmp_path: Path,
    raw_config: dict[str, Any],
) -> Path:
    """Write modified configuration data to a temporary YAML file."""
    config_path = tmp_path / "portfolio.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            raw_config,
            file,
            sort_keys=False,
        )

    return config_path


def test_current_portfolio_weights_sum_to_one() -> None:
    # Portfolio weights must represent a fully invested allocation
    config = load_portfolio_config(CONFIG_PATH)

    snapshot = get_portfolio_snapshot(
        config=config,
        portfolio_id="current",
    )

    total_weight = sum(snapshot.weights.values()) + snapshot.cash_weight

    assert total_weight == pytest.approx(1.0)


def test_pre_diversification_weights_sum_to_one() -> None:
    # Historical snapshots must satisfy the same allocation invariant
    config = load_portfolio_config(CONFIG_PATH)

    snapshot = get_portfolio_snapshot(
        config=config,
        portfolio_id="pre_diversification",
    )

    total_weight = sum(snapshot.weights.values()) + snapshot.cash_weight

    assert total_weight == pytest.approx(1.0)


def test_data_universe_contains_required_assets() -> None:
    # The shared universe must support both current and historical snapshots
    config = load_portfolio_config(CONFIG_PATH)

    assert len(config.assets) == 19
    assert config.assets["amd"].ticker == "AMD"
    assert config.assets["cameco"].ticker == "CCO.TO"


def test_current_weights_exclude_amd() -> None:
    # Assets in the data universe should appear only when held by the snapshot
    config = load_portfolio_config(CONFIG_PATH)

    weights = get_portfolio_weights(
        config=config,
        portfolio_id="current",
    )

    assert "AMD" not in weights
    assert weights["CSH2"] == pytest.approx(0.075)


def test_stress_group_adapts_to_snapshot() -> None:
    # Stress groups should resolve from the holdings active in each snapshot
    config = load_portfolio_config(CONFIG_PATH)

    current_groups = resolve_correlation_stress_groups(
        config=config,
        portfolio_id="current",
    )
    historical_groups = resolve_correlation_stress_groups(
        config=config,
        portfolio_id="pre_diversification",
    )

    assert "AMD" not in current_groups["technology_growth"]
    assert "AMD" in historical_groups["technology_growth"]
    assert len(current_groups["technology_growth"]) == 5
    assert len(historical_groups["technology_growth"]) == 6


def test_zero_weight_asset_does_not_activate_stress_group(
    tmp_path: Path,
) -> None:
    # Zero-weight positions carry no exposure and should remain inactive
    raw_config = _load_raw_config()
    raw_config["portfolio"]["portfolios"]["current"]["weights"]["amd"] = 0.0

    config_path = _write_test_config(
        tmp_path=tmp_path,
        raw_config=raw_config,
    )
    config = load_portfolio_config(config_path)

    groups = resolve_correlation_stress_groups(
        config=config,
        portfolio_id="current",
    )

    assert "AMD" not in groups["technology_growth"]


def test_portfolio_base_currency_is_gbp() -> None:
    # The case-study configuration should preserve GBP as its reporting currency
    config = load_portfolio_config(CONFIG_PATH)

    assert config.base_currency == "GBP"


def test_unknown_portfolio_snapshot_raises_value_error() -> None:
    # Invalid snapshot identifiers should fail before downstream analysis begins
    config = load_portfolio_config(CONFIG_PATH)

    with pytest.raises(
        ValueError,
        match="Unknown portfolio snapshot",
    ):
        get_portfolio_snapshot(
            config=config,
            portfolio_id="unknown",
        )


def test_estimation_start_date_must_precede_end_date(
    tmp_path: Path,
) -> None:
    # An invalid estimation window would make historical analysis ill-defined
    raw_config = _load_raw_config()
    estimation = raw_config["portfolio"]["estimation"]
    estimation["start_date"] = "2026-09-09"
    estimation["end_date"] = "2021-09-09"

    config_path = _write_test_config(
        tmp_path=tmp_path,
        raw_config=raw_config,
    )

    with pytest.raises(
        ValueError,
        match="Estimation start date must precede end date",
    ):
        load_portfolio_config(config_path)


def test_asset_display_names_must_be_unique(
    tmp_path: Path,
) -> None:
    # Duplicate display names would create ambiguous labels in pandas outputs
    raw_config = _load_raw_config()
    raw_config["portfolio"]["assets"]["amd"]["name"] = "Micron"

    config_path = _write_test_config(
        tmp_path=tmp_path,
        raw_config=raw_config,
    )

    with pytest.raises(
        ValueError,
        match="Asset display names must be unique",
    ):
        load_portfolio_config(config_path)


def test_proxy_must_reference_configured_asset(
    tmp_path: Path,
) -> None:
    # Proxy data should only extend assets already defined in the universe
    raw_config = _load_raw_config()
    raw_config["portfolio"]["proxies"]["unknown_asset"] = {
        "provider": "Bank of England",
        "series_code": "TEST",
        "apply_before_first_observation": True,
    }

    config_path = _write_test_config(
        tmp_path=tmp_path,
        raw_config=raw_config,
    )

    with pytest.raises(
        ValueError,
        match="Historical proxies reference unknown assets",
    ):
        load_portfolio_config(config_path)


def test_duplicate_assets_in_stress_group_are_rejected(
    tmp_path: Path,
) -> None:
    # Repeated group members would duplicate pair definitions in stress analysis
    raw_config = _load_raw_config()
    group = raw_config["portfolio"]["correlation_stress"]["groups"]["technology_growth"]
    group.append("asml")

    config_path = _write_test_config(
        tmp_path=tmp_path,
        raw_config=raw_config,
    )

    with pytest.raises(
        ValueError,
        match="contains duplicate assets",
    ):
        load_portfolio_config(config_path)


def test_negative_correlation_stress_target_is_valid(
    tmp_path: Path,
) -> None:
    # Empirical calibration may produce valid correlation floors below zero
    raw_config = _load_raw_config()
    raw_config["portfolio"]["correlation_stress"]["target_correlations"] = [
        -0.20,
        0.20,
        0.60,
    ]

    config_path = _write_test_config(
        tmp_path=tmp_path,
        raw_config=raw_config,
    )

    config = load_portfolio_config(config_path)

    assert config.correlation_stress is not None
    assert config.correlation_stress.target_correlations == (
        -0.20,
        0.20,
        0.60,
    )


def test_perfect_correlation_stress_target_is_rejected(
    tmp_path: Path,
) -> None:
    # A target of one would conflict with downstream positive-definite matrices
    raw_config = _load_raw_config()
    raw_config["portfolio"]["correlation_stress"]["target_correlations"] = [
        0.60,
        0.80,
        1.00,
    ]

    config_path = _write_test_config(
        tmp_path=tmp_path,
        raw_config=raw_config,
    )

    with pytest.raises(
        ValueError,
        match="must lie in the interval",
    ):
        load_portfolio_config(config_path)


def test_correlation_stress_targets_must_be_increasing(
    tmp_path: Path,
) -> None:
    # Ordered targets keep scenario outputs consistent from mild to severe stress
    raw_config = _load_raw_config()
    raw_config["portfolio"]["correlation_stress"]["target_correlations"] = [
        0.60,
        0.85,
        0.80,
    ]

    config_path = _write_test_config(
        tmp_path=tmp_path,
        raw_config=raw_config,
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        load_portfolio_config(config_path)
