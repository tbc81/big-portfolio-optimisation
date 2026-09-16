"""Download and apply historical proxy series to incomplete asset prices."""

from dataclasses import dataclass
from datetime import date
from io import StringIO
from math import isfinite

import pandas as pd
import requests

BOE_DATABASE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
)
REQUEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class ProxyBackfillResult:
    """Store prices and diagnostics from a proxy backfill."""

    prices: pd.DataFrame
    first_observed_date: pd.Timestamp
    scale_factor: float
    proxy_observations: int


def download_boe_series(
    series_code: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Download a Bank of England time series."""
    parameters = {
        "csv.x": "yes",
        "Datefrom": _format_boe_date(start_date),
        "Dateto": _format_boe_date(end_date),
        "SeriesCodes": series_code,
        "CSVF": "TN",
        "UsingCodes": "Y",
        "VPD": "Y",
        "VFD": "N",
    }

    response = requests.get(
        BOE_DATABASE_URL,
        params=parameters,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = pd.read_csv(StringIO(response.text))
    required_columns = {"DATE", series_code}

    if not required_columns.issubset(data.columns):
        raise RuntimeError("Unexpected Bank of England response.")

    data["DATE"] = pd.to_datetime(
        data["DATE"],
        dayfirst=True,
        errors="raise",
    )

    series = data.set_index("DATE")[series_code]
    series = pd.to_numeric(series, errors="raise").sort_index()

    if series.empty or series.dropna().empty:
        raise RuntimeError(f"No data returned for '{series_code}'.")

    if series.index.has_duplicates:
        raise RuntimeError(f"Duplicate dates returned for '{series_code}'.")

    return series


def backfill_price_with_proxy(
    prices: pd.DataFrame,
    asset_name: str,
    proxy_index: pd.Series,
) -> ProxyBackfillResult:
    """Backfill an asset's pre-history with a scaled proxy index."""
    if asset_name not in prices.columns:
        raise ValueError(f"Asset '{asset_name}' is missing from price data.")

    _validate_time_index(
        index=prices.index,
        name="Price",
    )
    _validate_time_index(
        index=proxy_index.index,
        name="Proxy",
    )

    first_observed_date = prices[asset_name].first_valid_index()

    if first_observed_date is None:
        raise ValueError(f"No observed prices found for '{asset_name}'.")

    # Forward alignment uses only proxy observations available on or before each date
    aligned_proxy = proxy_index.reindex(
        prices.index,
        method="ffill",
    )

    first_proxy_value = aligned_proxy.loc[first_observed_date]

    if pd.isna(first_proxy_value):
        raise ValueError("Proxy data is unavailable at the first observed asset date.")

    first_asset_price = float(prices.loc[first_observed_date, asset_name])
    first_proxy_value = float(first_proxy_value)

    if not isfinite(first_asset_price) or first_asset_price <= 0.0:
        raise ValueError("First observed asset price must be finite and positive.")

    if not isfinite(first_proxy_value) or first_proxy_value <= 0.0:
        raise ValueError("Proxy value at the join date must be finite and positive.")

    # Scale = asset price at the join date / proxy value at the join date
    scale_factor = first_asset_price / first_proxy_value
    scaled_proxy = aligned_proxy * scale_factor

    # Preserve every observed asset price from the first market observation onward
    pre_history = prices.index < first_observed_date

    if scaled_proxy.loc[pre_history].isna().any():
        raise ValueError("Proxy data does not cover the required pre-history.")

    backfilled_prices = prices.copy()
    backfilled_prices.loc[pre_history, asset_name] = scaled_proxy.loc[pre_history]

    return ProxyBackfillResult(
        prices=backfilled_prices,
        first_observed_date=first_observed_date,
        scale_factor=scale_factor,
        proxy_observations=int(pre_history.sum()),
    )


def _validate_time_index(
    index: pd.Index,
    name: str,
) -> None:
    """Validate a time-series index used for historical alignment."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} data must use a DatetimeIndex.")

    if index.has_duplicates:
        raise ValueError(f"{name} data contains duplicate dates.")

    # Temporal filling depends on observations being ordered chronologically
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} data must be sorted by date.")


def _format_boe_date(date_string: str) -> str:
    """Convert an ISO date into the Bank of England request format."""
    parsed_date = date.fromisoformat(date_string)

    return parsed_date.strftime("%d/%b/%Y")
