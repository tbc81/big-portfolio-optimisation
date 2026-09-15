from collections.abc import (
    Iterable,
)

import pandas as pd
import yfinance as yf

from big_portfolio.config import AssetConfig

FX_TICKERS = {
    "EUR": "EURGBP=X",
    "USD": "GBPUSD=X",
    "CAD": "GBPCAD=X",
}


# Asset price extraction
def download_adjusted_prices(
    assets: dict[str, AssetConfig],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download adjusted daily closing prices for portfolio assets."""
    ticker_to_name = {asset.ticker: asset.name for asset in assets.values()}

    # Passes tickers to download function
    prices = _download_close_prices(
        tickers=ticker_to_name.keys(),
        start_date=start_date,
        end_date=end_date,
    )
    # renames columns of returned DataFrame to asset names
    prices = prices.rename(columns=ticker_to_name)

    # Validates that all expected assets have been downloaded
    expected_assets = set(ticker_to_name.values())

    missing_assets = expected_assets - set(prices.columns)

    if missing_assets:
        raise RuntimeError(f"Failed to download prices for: {sorted(missing_assets)}")

    # Preserve portfolio order regardless of Yahoo's returned column order
    return prices.loc[
        :,
        list(ticker_to_name.values()),
    ]


# Currency FX rate extraction
def download_fx_rates(
    currencies: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download GBP FX rates required by the portfolio."""

    # Converts currency list into unique set and removes GBP and GBX for efficiency
    required_currencies = set(currencies) - {"GBP", "GBX"}

    # Validate currencies before constructing Yahoo FX tickers
    unsupported = required_currencies - set(FX_TICKERS)

    if unsupported:
        raise ValueError(f"Unsupported currencies: {sorted(unsupported)}")

    # Inverted lookup dictionary to map FX tickers back to corresponding currency codes
    ticker_to_currency = {
        FX_TICKERS[currency]: currency for currency in sorted(required_currencies)
    }
    # A GBP-only portfolio requires no FX data
    if not ticker_to_currency:
        return pd.DataFrame()

    # Yahoo may return columns in a different order from the request
    fx_rates = _download_close_prices(
        tickers=ticker_to_currency.keys(),
        start_date=start_date,
        end_date=end_date,
    )

    fx_rates = fx_rates.rename(columns=ticker_to_currency)

    return fx_rates.loc[
        :,
        sorted(required_currencies),
    ]


# Helper function to download adjusted closing prices
def _download_close_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download adjusted closing prices from Yahoo Finance."""
    ticker_list = list(tickers)

    market_data = yf.download(
        tickers=ticker_list,
        start=start_date,
        end=end_date,
        interval="1d",  # daily frequency
        auto_adjust=True,  # Include splits and dividend adjustments
        actions=False,  # excludes corporate actions from the returned data
        progress=False,  # disables progress bar to avoid cluttering output
        threads=True,  # improved performance when fetching data for multiple tickers
    )

    # Verifies returned DataFrame is not empty
    if market_data.empty:
        raise RuntimeError("Yahoo Finance returned no market data.")

    # Extracts adjusted closing prices.
    prices = market_data["Close"].copy()

    # Keep a consistent DataFrame return type for a single ticker
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=ticker_list[0])

    # Return observations in chronological order
    return prices.sort_index()
