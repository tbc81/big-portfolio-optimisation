"""Test market-data downloads, FX mapping and Yahoo recovery behaviour."""

import numpy as np
import pandas as pd
import pytest

from big_portfolio.config import AssetConfig
from big_portfolio.data import market_data


def test_adjusted_prices_preserve_configured_asset_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Output columns should follow configuration order for stable downstream matrices
    assets = {
        "second": AssetConfig(
            name="Second Asset",
            ticker="BBB",
            currency="GBP",
        ),
        "first": AssetConfig(
            name="First Asset",
            ticker="AAA",
            currency="GBP",
        ),
    }

    dates = pd.to_datetime(["2026-01-02"])

    def fake_download(
        tickers: object,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "AAA": [100.0],
                "BBB": [200.0],
            },
            index=dates,
        )

    monkeypatch.setattr(
        market_data,
        "_download_close_prices",
        fake_download,
    )

    prices = market_data.download_adjusted_prices(
        assets=assets,
        start_date="2026-01-01",
        end_date="2026-01-03",
    )

    assert list(prices.columns) == [
        "Second Asset",
        "First Asset",
    ]


def test_fx_rates_use_configured_yahoo_tickers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Currency codes should map to the Yahoo symbols expected by GBP conversion
    captured_tickers: list[str] = []
    dates = pd.to_datetime(["2026-01-02"])

    def fake_download(
        tickers: object,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        captured_tickers.extend(tickers)

        return pd.DataFrame(
            {
                "GBPCAD=X": [1.80],
                "EURGBP=X": [0.85],
                "GBPUSD=X": [1.30],
            },
            index=dates,
        )

    monkeypatch.setattr(
        market_data,
        "_download_close_prices",
        fake_download,
    )

    fx_rates = market_data.download_fx_rates(
        currencies=["USD", "GBP", "EUR", "CAD", "GBX"],
        start_date="2026-01-01",
        end_date="2026-01-03",
    )

    assert captured_tickers == [
        "GBPCAD=X",
        "EURGBP=X",
        "GBPUSD=X",
    ]
    assert list(fx_rates.columns) == [
        "CAD",
        "EUR",
        "USD",
    ]


def test_sterling_assets_require_no_fx_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GBP and GBX positions should avoid unnecessary external FX requests
    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("FX download should not be called.")

    monkeypatch.setattr(
        market_data,
        "_download_close_prices",
        fail_if_called,
    )

    fx_rates = market_data.download_fx_rates(
        currencies=["GBP", "GBX"],
        start_date="2026-01-01",
        end_date="2026-01-03",
    )

    assert fx_rates.empty


def test_failed_bulk_ticker_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A ticker missing from a bulk response should receive an individual retry
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
        ]
    )

    calls: list[list[str]] = []

    def fake_request(
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        calls.append(tickers)

        if len(tickers) > 1:
            return pd.DataFrame(
                {
                    "AAA": [
                        100.0,
                        101.0,
                    ],
                    "BBB": [
                        np.nan,
                        np.nan,
                    ],
                },
                index=dates,
            )

        return pd.DataFrame(
            {
                "BBB": [
                    200.0,
                    202.0,
                ]
            },
            index=dates,
        )

    monkeypatch.setattr(
        market_data,
        "_request_close_prices",
        fake_request,
    )

    prices = market_data._download_close_prices(
        tickers=[
            "AAA",
            "BBB",
        ],
        start_date="2026-01-01",
        end_date="2026-01-06",
    )

    assert calls == [
        ["AAA", "BBB"],
        ["BBB"],
    ]
    assert not prices["BBB"].isna().any()


def test_recovered_ticker_keeps_deterministic_column_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Retry joins must not reorder assets used by later portfolio calculations
    dates = pd.to_datetime(["2026-01-02"])

    def fake_request(
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if len(tickers) > 1:
            return pd.DataFrame(
                {
                    "AAA": [100.0],
                    "BBB": [np.nan],
                },
                index=dates,
            )

        return pd.DataFrame(
            {
                "BBB": [200.0],
            },
            index=dates,
        )

    monkeypatch.setattr(
        market_data,
        "_request_close_prices",
        fake_request,
    )

    prices = market_data._download_close_prices(
        tickers=["BBB", "AAA"],
        start_date="2026-01-01",
        end_date="2026-01-03",
    )

    assert list(prices.columns) == [
        "BBB",
        "AAA",
    ]


def test_persistent_download_failure_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unresolved symbols should fail clearly before incomplete data reaches estimation
    dates = pd.to_datetime(["2026-01-02"])

    def fake_request(
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {ticker: [np.nan] for ticker in tickers},
            index=dates,
        )

    monkeypatch.setattr(
        market_data,
        "_request_close_prices",
        fake_request,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to download usable price history",
    ):
        market_data._download_close_prices(
            tickers=["AAA"],
            start_date="2026-01-01",
            end_date="2026-01-06",
        )


def test_persistent_failure_uses_configured_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Recovery should make one bulk request followed by the configured retries
    dates = pd.to_datetime(["2026-01-02"])
    calls = 0

    def fake_request(
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        nonlocal calls
        calls += 1

        return pd.DataFrame(
            {ticker: [np.nan] for ticker in tickers},
            index=dates,
        )

    monkeypatch.setattr(
        market_data,
        "_request_close_prices",
        fake_request,
    )

    with pytest.raises(RuntimeError):
        market_data._download_close_prices(
            tickers=["AAA"],
            start_date="2026-01-01",
            end_date="2026-01-03",
        )

    assert calls == 1 + market_data.INDIVIDUAL_RETRY_ATTEMPTS


def test_duplicate_ticker_request_raises_error() -> None:
    # Duplicate symbols would make the returned column mapping ambiguous
    with pytest.raises(
        ValueError,
        match="Ticker requests must be unique",
    ):
        market_data._download_close_prices(
            tickers=["AAA", "AAA"],
            start_date="2026-01-01",
            end_date="2026-01-03",
        )


def test_single_ticker_response_is_normalised_to_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Single-symbol Yahoo output should match the DataFrame shape used by bulk calls
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
        ]
    )

    market_response = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
            ]
        },
        index=dates,
    )

    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda **kwargs: market_response,
    )

    prices = market_data._request_close_prices(
        tickers=["AAA"],
        start_date="2026-01-01",
        end_date="2026-01-06",
    )

    assert isinstance(prices, pd.DataFrame)
    assert list(prices.columns) == ["AAA"]
    np.testing.assert_allclose(
        prices["AAA"].to_numpy(),
        np.array([100.0, 101.0]),
    )


def test_unsupported_fx_currency_raises_error() -> None:
    # Currency support should be explicit before a Yahoo symbol is requested
    with pytest.raises(
        ValueError,
        match="Unsupported currencies",
    ):
        market_data.download_fx_rates(
            currencies=["JPY"],
            start_date="2026-01-01",
            end_date="2026-01-03",
        )
