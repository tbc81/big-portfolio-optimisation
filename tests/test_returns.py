"""Test weekly resampling and complete log-return scenario construction."""

import numpy as np
import pandas as pd
import pytest

from big_portfolio.data.returns import build_weekly_log_returns


def test_weekly_resampling_uses_last_price() -> None:
    # Each weekly scenario should use the final available market price in that period
    dates = pd.to_datetime(
        [
            "2026-01-05",
            "2026-01-06",
            "2026-01-09",
            "2026-01-12",
            "2026-01-16",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                102.0,
                105.0,
                107.0,
                110.0,
            ]
        },
        index=dates,
    )

    result = build_weekly_log_returns(
        prices=prices,
        frequency="W-FRI",
    )

    expected_weekly_prices = np.array(
        [
            105.0,
            110.0,
        ]
    )

    np.testing.assert_allclose(
        result.weekly_prices["Asset A"].to_numpy(),
        expected_weekly_prices,
    )


def test_log_returns_are_calculated_correctly() -> None:
    # Weekly returns should follow r_t = ln(P_t / P_(t-1))
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-09",
            "2026-01-16",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                110.0,
                121.0,
            ]
        },
        index=dates,
    )

    result = build_weekly_log_returns(
        prices=prices,
    )

    expected_return = np.log(1.10)
    expected = np.array(
        [
            expected_return,
            expected_return,
        ]
    )

    np.testing.assert_allclose(
        result.log_returns["Asset A"].to_numpy(),
        expected,
    )


def test_first_return_scenario_is_dropped() -> None:
    # The first weekly price has no preceding observation for a return calculation
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-09",
            "2026-01-16",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                101.0,
                102.0,
            ]
        },
        index=dates,
    )

    result = build_weekly_log_returns(
        prices=prices,
    )

    assert len(result.weekly_prices) == 3
    assert len(result.log_returns) == 2
    assert result.dropped_scenarios == 1


def test_incomplete_scenarios_are_dropped() -> None:
    # Covariance estimation should receive only jointly observed asset-return scenarios
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-09",
            "2026-01-16",
            "2026-01-23",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                105.0,
                110.0,
                115.0,
            ],
            "Asset B": [
                200.0,
                210.0,
                np.nan,
                220.0,
            ],
        },
        index=dates,
    )

    result = build_weekly_log_returns(
        prices=prices,
    )

    assert not result.log_returns.isna().any().any()
    assert len(result.log_returns) == 1
    assert result.dropped_scenarios == 3


def test_non_positive_price_raises_error() -> None:
    # Log returns require every observed market price to be strictly positive
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-09",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                0.0,
            ]
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        build_weekly_log_returns(
            prices=prices,
        )


def test_non_finite_price_raises_error() -> None:
    # Infinite observations would produce invalid numerical return scenarios
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-09",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                np.inf,
            ]
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="Observed prices must be finite",
    ):
        build_weekly_log_returns(
            prices=prices,
        )


def test_missing_asset_history_raises_error() -> None:
    # Every asset needs observed history before joint return scenarios can be formed
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-09",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                101.0,
            ],
            "Asset B": [
                np.nan,
                np.nan,
            ],
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="No usable price history",
    ):
        build_weekly_log_returns(
            prices=prices,
        )


def test_unsorted_price_index_raises_error() -> None:
    # Weekly resampling should operate on observations ordered chronologically
    dates = pd.to_datetime(
        [
            "2026-01-09",
            "2026-01-02",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                101.0,
                100.0,
            ]
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="Price data must be sorted by date",
    ):
        build_weekly_log_returns(
            prices=prices,
        )


def test_duplicate_price_dates_raise_error() -> None:
    # Multiple observations for one date would make the daily time series ambiguous
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-02",
            "2026-01-09",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                101.0,
                102.0,
            ]
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="duplicate dates",
    ):
        build_weekly_log_returns(
            prices=prices,
        )


def test_non_numeric_price_column_raises_error() -> None:
    # Asset-price columns must contain numerical observations for return arithmetic
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-09",
        ]
    )

    prices = pd.DataFrame(
        {
            "Asset A": [
                "100.0",
                "110.0",
            ]
        },
        index=dates,
    )

    with pytest.raises(
        TypeError,
        match="Price data must be numeric",
    ):
        build_weekly_log_returns(
            prices=prices,
        )


def test_non_datetime_index_raises_error() -> None:
    # Resampling requires a DatetimeIndex to define the weekly periods
    prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                110.0,
            ]
        }
    )

    with pytest.raises(
        TypeError,
        match="DatetimeIndex",
    ):
        build_weekly_log_returns(
            prices=prices,
        )
