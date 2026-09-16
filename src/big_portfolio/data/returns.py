"""Construct complete weekly return scenarios from daily asset prices."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass(frozen=True)
class ReturnScenarioResult:
    """Store resampled prices and complete weekly return scenarios."""

    weekly_prices: pd.DataFrame
    log_returns: pd.DataFrame
    dropped_scenarios: int


def build_weekly_log_returns(
    prices: pd.DataFrame,
    frequency: str = "W-FRI",
) -> ReturnScenarioResult:
    """Construct complete weekly log-return scenarios from daily prices."""
    _validate_prices(prices)

    # Use the last available market observation in each weekly period
    weekly_prices = prices.resample(frequency).last()

    # Weekly log return: r_t = ln(P_t / P_(t-1))
    raw_log_returns = np.log(weekly_prices / weekly_prices.shift(1))

    # Covariance estimation requires all asset returns in the same scenario
    complete_log_returns = raw_log_returns.dropna(how="any")

    if complete_log_returns.empty:
        raise ValueError("No complete return scenarios were produced.")

    dropped_scenarios = len(raw_log_returns) - len(complete_log_returns)

    return ReturnScenarioResult(
        weekly_prices=weekly_prices,
        log_returns=complete_log_returns,
        dropped_scenarios=dropped_scenarios,
    )


def _validate_prices(prices: pd.DataFrame) -> None:
    """Validate daily price data before return construction."""
    if prices.empty:
        raise ValueError("Price data cannot be empty.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price data must use a DatetimeIndex.")

    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")

    # Resampling assumes observations are ordered chronologically
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price data must be sorted by date.")

    if prices.columns.has_duplicates:
        raise ValueError("Price data contains duplicate asset columns.")

    non_numeric_columns = [
        column
        for column in prices.columns
        if not is_numeric_dtype(prices[column].dtype)
    ]

    if non_numeric_columns:
        raise TypeError(
            f"Price data must be numeric for: {sorted(non_numeric_columns)}"
        )

    missing_histories = [
        column for column in prices.columns if prices[column].dropna().empty
    ]

    if missing_histories:
        raise ValueError(f"No usable price history for: {sorted(missing_histories)}")

    values = prices.to_numpy(dtype=float)
    observed_values = values[~np.isnan(values)]

    if not np.isfinite(observed_values).all():
        raise ValueError("Observed prices must be finite.")

    if (observed_values <= 0.0).any():
        raise ValueError("Observed prices must be strictly positive.")
