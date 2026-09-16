"""Download asset prices and GBP foreign-exchange rates from Yahoo Finance."""

from collections.abc import Iterable

import pandas as pd
import yfinance as yf

from big_portfolio.config import AssetConfig

# Yahoo symbols follow the GBP quote conventions handled in currency.py
FX_TICKERS = {
    "EUR": "EURGBP=X",
    "USD": "GBPUSD=X",
    "CAD": "GBPCAD=X",
}

INDIVIDUAL_RETRY_ATTEMPTS = 3


def download_adjusted_prices(
    assets: dict[str, AssetConfig],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download adjusted daily closing prices for configured assets."""
    ticker_to_name = {asset.ticker: asset.name for asset in assets.values()}

    prices = _download_close_prices(
        tickers=ticker_to_name,
        start_date=start_date,
        end_date=end_date,
    )
    prices = prices.rename(columns=ticker_to_name)

    expected_assets = list(ticker_to_name.values())
    missing_assets = set(expected_assets) - set(prices.columns)

    if missing_assets:
        raise RuntimeError(f"Failed to download prices for: {sorted(missing_assets)}")

    # Preserve configuration order independently of Yahoo's returned column order
    return prices.loc[:, expected_assets]


def download_fx_rates(
    currencies: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download Yahoo FX rates required for GBP price conversion."""
    required_currencies = set(currencies) - {"GBP", "GBX"}
    unsupported_currencies = required_currencies - set(FX_TICKERS)

    if unsupported_currencies:
        raise ValueError(f"Unsupported currencies: {sorted(unsupported_currencies)}")

    ordered_currencies = sorted(required_currencies)

    # Sterling-only portfolios do not require an external FX time series
    if not ordered_currencies:
        return pd.DataFrame()

    ticker_to_currency = {
        FX_TICKERS[currency]: currency for currency in ordered_currencies
    }

    fx_rates = _download_close_prices(
        tickers=ticker_to_currency,
        start_date=start_date,
        end_date=end_date,
    )
    fx_rates = fx_rates.rename(columns=ticker_to_currency)

    return fx_rates.loc[:, ordered_currencies]


def _download_close_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download closing prices with individual recovery for failed tickers."""
    ticker_list = list(tickers)

    if not ticker_list:
        raise ValueError("At least one ticker is required.")

    if len(ticker_list) != len(set(ticker_list)):
        raise ValueError("Ticker requests must be unique.")

    prices = _request_close_prices(
        tickers=ticker_list,
        start_date=start_date,
        end_date=end_date,
    )

    failed_tickers = [
        ticker
        for ticker in ticker_list
        if not _has_usable_price_history(
            prices=prices,
            ticker=ticker,
        )
    ]

    for ticker in failed_tickers:
        retry_prices = pd.DataFrame()

        # Individual requests can recover symbols omitted from a bulk Yahoo response
        for _ in range(INDIVIDUAL_RETRY_ATTEMPTS):
            retry_prices = _request_close_prices(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
            )

            if _has_usable_price_history(
                prices=retry_prices,
                ticker=ticker,
            ):
                break

        if not _has_usable_price_history(
            prices=retry_prices,
            ticker=ticker,
        ):
            continue

        if prices.empty:
            prices = retry_prices[[ticker]].copy()
            continue

        prices = prices.drop(
            columns=[ticker],
            errors="ignore",
        ).join(
            retry_prices[[ticker]],
            how="outer",
        )

    unresolved_tickers = [
        ticker
        for ticker in ticker_list
        if not _has_usable_price_history(
            prices=prices,
            ticker=ticker,
        )
    ]

    if unresolved_tickers:
        raise RuntimeError(
            f"Failed to download usable price history for: {sorted(unresolved_tickers)}"
        )

    # Return a deterministic column order for downstream portfolio calculations
    return prices.loc[:, ticker_list].sort_index()


def _request_close_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Make one Yahoo Finance adjusted-price request."""
    # auto_adjust applies split and dividend adjustments used for return estimation
    market_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
    )

    if market_data.empty:
        return pd.DataFrame()

    if "Close" not in market_data.columns:
        raise RuntimeError("Yahoo Finance response does not contain closing prices.")

    prices = market_data["Close"].copy()

    # Normalise single-ticker output to the same DataFrame shape as bulk requests
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    elif len(tickers) == 1 and prices.shape[1] == 1:
        prices.columns = tickers

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise RuntimeError("Yahoo Finance returned an invalid date index.")

    if prices.index.has_duplicates:
        raise RuntimeError("Yahoo Finance returned duplicate price dates.")

    return prices.sort_index()


def _has_usable_price_history(
    prices: pd.DataFrame,
    ticker: str,
) -> bool:
    """Return whether a ticker contains at least one observed price."""
    return ticker in prices.columns and not prices[ticker].dropna().empty
