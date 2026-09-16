"""Test local-currency price conversion into GBP."""

import numpy as np
import pandas as pd
import pytest

from big_portfolio.config import AssetConfig
from big_portfolio.data.currency import convert_prices_to_gbp


@pytest.fixture
def assets() -> dict[str, AssetConfig]:
    """Return assets covering every supported currency convention."""
    return {
        "gbp_asset": AssetConfig(
            name="GBP Asset",
            ticker="GBP.TEST",
            currency="GBP",
        ),
        "gbx_asset": AssetConfig(
            name="GBX Asset",
            ticker="GBX.TEST",
            currency="GBX",
        ),
        "eur_asset": AssetConfig(
            name="EUR Asset",
            ticker="EUR.TEST",
            currency="EUR",
        ),
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
        ),
        "cad_asset": AssetConfig(
            name="CAD Asset",
            ticker="CAD.TEST",
            currency="CAD",
        ),
    }


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    """Return representative trading dates for currency tests."""
    return pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
        ]
    )


def test_currency_conversion(
    assets: dict[str, AssetConfig],
    dates: pd.DatetimeIndex,
) -> None:
    # Each supported quote convention should produce the correct GBP price
    prices = pd.DataFrame(
        {
            "GBP Asset": [10.0, 12.0],
            "GBX Asset": [250.0, 300.0],
            "EUR Asset": [100.0, 110.0],
            "USD Asset": [130.0, 140.0],
            "CAD Asset": [180.0, 190.0],
        },
        index=dates,
    )

    fx_rates = pd.DataFrame(
        {
            "EUR": [0.85, 0.86],
            "USD": [1.30, 1.40],
            "CAD": [1.80, 1.90],
        },
        index=dates,
    )

    converted = convert_prices_to_gbp(
        prices=prices,
        assets=assets,
        fx_rates=fx_rates,
    )

    expected = pd.DataFrame(
        {
            "GBP Asset": [10.0, 12.0],
            "GBX Asset": [2.5, 3.0],
            "EUR Asset": [85.0, 94.6],
            "USD Asset": [100.0, 100.0],
            "CAD Asset": [100.0, 100.0],
        },
        index=dates,
    )

    np.testing.assert_allclose(
        converted.to_numpy(),
        expected.to_numpy(),
    )


def test_fx_alignment_uses_latest_available_observation() -> None:
    # Asset dates should use the latest FX rate known on or before each date
    price_dates = pd.to_datetime(
        [
            "2026-01-03",
            "2026-01-05",
        ]
    )

    prices = pd.DataFrame(
        {
            "USD Asset": [130.0, 143.0],
        },
        index=price_dates,
    )

    assets = {
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
        )
    }

    fx_rates = pd.DataFrame(
        {
            "USD": [1.30, 1.40],
        },
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-04",
            ]
        ),
    )

    converted = convert_prices_to_gbp(
        prices=prices,
        assets=assets,
        fx_rates=fx_rates,
    )

    expected = np.array(
        [
            100.0,
            143.0 / 1.40,
        ]
    )

    np.testing.assert_allclose(
        converted["USD Asset"].to_numpy(),
        expected,
    )


def test_fx_alignment_does_not_use_future_observations() -> None:
    # Missing earlier FX history must fail before any future rate can be used
    prices = pd.DataFrame(
        {
            "USD Asset": [130.0, 140.0],
        },
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-05",
            ]
        ),
    )

    assets = {
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
        )
    }

    fx_rates = pd.DataFrame(
        {
            "USD": [1.30],
        },
        index=pd.to_datetime(["2026-01-03"]),
    )

    with pytest.raises(
        ValueError,
        match="missing or non-finite values after alignment",
    ):
        convert_prices_to_gbp(
            prices=prices,
            assets=assets,
            fx_rates=fx_rates,
        )


def test_gbp_and_gbx_assets_do_not_require_fx_rates(
    dates: pd.DatetimeIndex,
) -> None:
    # Sterling assets should convert correctly without an external FX series
    assets = {
        "gbp_asset": AssetConfig(
            name="GBP Asset",
            ticker="GBP.TEST",
            currency="GBP",
        ),
        "gbx_asset": AssetConfig(
            name="GBX Asset",
            ticker="GBX.TEST",
            currency="GBX",
        ),
    }

    prices = pd.DataFrame(
        {
            "GBP Asset": [10.0, 12.0],
            "GBX Asset": [250.0, 300.0],
        },
        index=dates,
    )

    converted = convert_prices_to_gbp(
        prices=prices,
        assets=assets,
        fx_rates=pd.DataFrame(),
    )

    np.testing.assert_allclose(
        converted["GBP Asset"].to_numpy(),
        np.array([10.0, 12.0]),
    )
    np.testing.assert_allclose(
        converted["GBX Asset"].to_numpy(),
        np.array([2.5, 3.0]),
    )


def test_missing_asset_price_column_raises_error(
    dates: pd.DatetimeIndex,
) -> None:
    # Every configured asset needs a corresponding local-price series
    assets = {
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
        )
    }

    prices = pd.DataFrame(
        {
            "Other Asset": [100.0, 101.0],
        },
        index=dates,
    )

    fx_rates = pd.DataFrame(
        {
            "USD": [1.30, 1.31],
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="Missing price data",
    ):
        convert_prices_to_gbp(
            prices=prices,
            assets=assets,
            fx_rates=fx_rates,
        )


def test_missing_fx_rate_raises_error(
    dates: pd.DatetimeIndex,
) -> None:
    # Foreign-currency assets require their configured GBP FX series
    assets = {
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
        )
    }

    prices = pd.DataFrame(
        {
            "USD Asset": [130.0, 140.0],
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="Missing FX rates",
    ):
        convert_prices_to_gbp(
            prices=prices,
            assets=assets,
            fx_rates=pd.DataFrame(index=dates),
        )


def test_non_positive_fx_rate_raises_error(
    dates: pd.DatetimeIndex,
) -> None:
    # Zero or negative FX rates would make currency conversion economically invalid
    assets = {
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
        )
    }

    prices = pd.DataFrame(
        {
            "USD Asset": [130.0, 140.0],
        },
        index=dates,
    )

    fx_rates = pd.DataFrame(
        {
            "USD": [1.30, 0.0],
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="FX rates must be positive",
    ):
        convert_prices_to_gbp(
            prices=prices,
            assets=assets,
            fx_rates=fx_rates,
        )


def test_unsorted_fx_index_raises_error(
    dates: pd.DatetimeIndex,
) -> None:
    # Historical FX alignment requires observations to be ordered through time
    assets = {
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
        )
    }

    prices = pd.DataFrame(
        {
            "USD Asset": [130.0, 140.0],
        },
        index=dates,
    )

    fx_rates = pd.DataFrame(
        {
            "USD": [1.40, 1.30],
        },
        index=pd.to_datetime(
            [
                "2026-01-05",
                "2026-01-02",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="FX data must be sorted by date",
    ):
        convert_prices_to_gbp(
            prices=prices,
            assets=assets,
            fx_rates=fx_rates,
        )


def test_unsupported_currency_raises_error(
    dates: pd.DatetimeIndex,
) -> None:
    # Currency conventions must be defined explicitly before they enter analysis
    assets = {
        "jpy_asset": AssetConfig(
            name="JPY Asset",
            ticker="JPY.TEST",
            currency="JPY",
        )
    }

    prices = pd.DataFrame(
        {
            "JPY Asset": [1000.0, 1100.0],
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported currencies",
    ):
        convert_prices_to_gbp(
            prices=prices,
            assets=assets,
            fx_rates=pd.DataFrame(index=dates),
        )
