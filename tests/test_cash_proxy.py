"""Test Bank of England proxy downloads and deterministic price backfilling."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import big_portfolio.data.cash_proxy as cash_proxy
from big_portfolio.data.cash_proxy import backfill_price_with_proxy


def test_download_boe_series_parses_numeric_time_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # External CSV data should become a chronological numeric pandas series
    response = Mock()
    response.text = "DATE,IUDZOS2\n02/01/2026,101.0\n01/01/2026,100.0\n"
    response.raise_for_status.return_value = None

    monkeypatch.setattr(
        cash_proxy.requests,
        "get",
        Mock(return_value=response),
    )

    series = cash_proxy.download_boe_series(
        series_code="IUDZOS2",
        start_date="2026-01-01",
        end_date="2026-01-02",
    )

    assert series.index.is_monotonic_increasing
    assert series.iloc[0] == pytest.approx(100.0)
    assert series.iloc[1] == pytest.approx(101.0)
    response.raise_for_status.assert_called_once_with()


def test_download_boe_series_rejects_duplicate_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Duplicate observations would make later historical alignment ambiguous
    response = Mock()
    response.text = "DATE,IUDZOS2\n01/01/2026,100.0\n01/01/2026,101.0\n"
    response.raise_for_status.return_value = None

    monkeypatch.setattr(
        cash_proxy.requests,
        "get",
        Mock(return_value=response),
    )

    with pytest.raises(
        RuntimeError,
        match="Duplicate dates",
    ):
        cash_proxy.download_boe_series(
            series_code="IUDZOS2",
            start_date="2026-01-01",
            end_date="2026-01-02",
        )


def test_proxy_backfills_only_pre_history() -> None:
    # Proxy values should fill only dates before the first observed market price
    dates = pd.date_range(
        "2026-01-01",
        periods=5,
        freq="D",
    )

    prices = pd.DataFrame(
        {
            "CSH2": [
                np.nan,
                np.nan,
                120.0,
                121.0,
                122.0,
            ]
        },
        index=dates,
    )

    proxy = pd.Series(
        [
            100.0,
            105.0,
            110.0,
            111.0,
            112.0,
        ],
        index=dates,
    )

    result = backfill_price_with_proxy(
        prices=prices,
        asset_name="CSH2",
        proxy_index=proxy,
    )

    expected_scale = 120.0 / 110.0

    assert result.scale_factor == pytest.approx(expected_scale)
    assert result.proxy_observations == 2
    assert result.first_observed_date == dates[2]

    assert result.prices.loc[
        dates[0],
        "CSH2",
    ] == pytest.approx(100.0 * expected_scale)

    assert result.prices.loc[
        dates[1],
        "CSH2",
    ] == pytest.approx(105.0 * expected_scale)

    assert result.prices.loc[
        dates[2],
        "CSH2",
    ] == pytest.approx(120.0)

    assert result.prices.loc[
        dates[4],
        "CSH2",
    ] == pytest.approx(122.0)


def test_proxy_joins_observed_price_continuously() -> None:
    # Scaling should make the proxy meet the first observed price at the join date
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    prices = pd.DataFrame(
        {
            "CSH2": [
                np.nan,
                150.0,
                151.0,
            ]
        },
        index=dates,
    )

    proxy = pd.Series(
        [
            95.0,
            100.0,
            101.0,
        ],
        index=dates,
    )

    result = backfill_price_with_proxy(
        prices=prices,
        asset_name="CSH2",
        proxy_index=proxy,
    )

    scaled_join_value = proxy.loc[dates[1]] * result.scale_factor

    assert scaled_join_value == pytest.approx(
        prices.loc[
            dates[1],
            "CSH2",
        ]
    )


def test_proxy_uses_latest_available_historical_observation() -> None:
    # Forward alignment should use the latest proxy value known by each asset date
    price_dates = pd.to_datetime(
        [
            "2026-01-03",
            "2026-01-05",
            "2026-01-07",
        ]
    )

    prices = pd.DataFrame(
        {
            "CSH2": [
                np.nan,
                120.0,
                121.0,
            ]
        },
        index=price_dates,
    )

    proxy = pd.Series(
        [
            100.0,
            104.0,
            106.0,
        ],
        index=pd.to_datetime(
            [
                "2026-01-01",
                "2026-01-04",
                "2026-01-06",
            ]
        ),
    )

    result = backfill_price_with_proxy(
        prices=prices,
        asset_name="CSH2",
        proxy_index=proxy,
    )

    expected_scale = 120.0 / 104.0

    assert result.prices.loc[
        price_dates[0],
        "CSH2",
    ] == pytest.approx(100.0 * expected_scale)


def test_observed_prices_are_not_replaced() -> None:
    # Real market observations must remain unchanged after proxy reconstruction
    dates = pd.date_range(
        "2026-01-01",
        periods=4,
        freq="D",
    )

    prices = pd.DataFrame(
        {
            "CSH2": [
                np.nan,
                100.0,
                105.0,
                110.0,
            ]
        },
        index=dates,
    )

    proxy = pd.Series(
        [
            90.0,
            95.0,
            120.0,
            130.0,
        ],
        index=dates,
    )

    result = backfill_price_with_proxy(
        prices=prices,
        asset_name="CSH2",
        proxy_index=proxy,
    )

    np.testing.assert_allclose(
        result.prices.loc[
            dates[1] :,
            "CSH2",
        ].to_numpy(),
        np.array(
            [
                100.0,
                105.0,
                110.0,
            ]
        ),
    )


def test_missing_asset_raises_error() -> None:
    # A proxy cannot be applied to an asset absent from the supplied price data
    prices = pd.DataFrame(
        {
            "Gold": [100.0],
        },
        index=pd.to_datetime(["2026-01-01"]),
    )

    proxy = pd.Series(
        [100.0],
        index=prices.index,
    )

    with pytest.raises(
        ValueError,
        match="missing from price data",
    ):
        backfill_price_with_proxy(
            prices=prices,
            asset_name="CSH2",
            proxy_index=proxy,
        )


def test_missing_observed_history_raises_error() -> None:
    # Scaling requires at least one genuine asset price to define the join point
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    prices = pd.DataFrame(
        {
            "CSH2": [
                np.nan,
                np.nan,
                np.nan,
            ]
        },
        index=dates,
    )

    proxy = pd.Series(
        [
            100.0,
            101.0,
            102.0,
        ],
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="No observed prices",
    ):
        backfill_price_with_proxy(
            prices=prices,
            asset_name="CSH2",
            proxy_index=proxy,
        )


def test_proxy_must_cover_required_pre_history() -> None:
    # Every reconstructed date needs a proxy observation available by that date
    dates = pd.date_range(
        "2026-01-01",
        periods=4,
        freq="D",
    )

    prices = pd.DataFrame(
        {
            "CSH2": [
                np.nan,
                np.nan,
                120.0,
                121.0,
            ]
        },
        index=dates,
    )

    proxy = pd.Series(
        [
            110.0,
            111.0,
        ],
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-03",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not cover the required pre-history",
    ):
        backfill_price_with_proxy(
            prices=prices,
            asset_name="CSH2",
            proxy_index=proxy,
        )


def test_unsorted_proxy_index_raises_error() -> None:
    # Temporal forward alignment depends on chronologically ordered observations
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    prices = pd.DataFrame(
        {
            "CSH2": [
                np.nan,
                120.0,
                121.0,
            ]
        },
        index=dates,
    )

    proxy = pd.Series(
        [
            101.0,
            100.0,
            102.0,
        ],
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-01",
                "2026-01-03",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="Proxy data must be sorted by date",
    ):
        backfill_price_with_proxy(
            prices=prices,
            asset_name="CSH2",
            proxy_index=proxy,
        )


def test_zero_proxy_value_at_join_date_raises_error() -> None:
    # The scaling ratio is undefined when the proxy join value is zero
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    prices = pd.DataFrame(
        {
            "CSH2": [
                np.nan,
                120.0,
                121.0,
            ]
        },
        index=dates,
    )

    proxy = pd.Series(
        [
            100.0,
            0.0,
            101.0,
        ],
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="Proxy value at the join date must be finite and positive",
    ):
        backfill_price_with_proxy(
            prices=prices,
            asset_name="CSH2",
            proxy_index=proxy,
        )
