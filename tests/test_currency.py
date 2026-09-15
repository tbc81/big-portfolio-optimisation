import numpy as np
import pandas as pd
import pytest

from big_portfolio.config import AssetConfig
from big_portfolio.data.currency import (
    convert_prices_to_gbp,
)


@pytest.fixture
def assets() -> dict[str, AssetConfig]:
    return {
        "gbp_asset": AssetConfig(
            name="GBP Asset",
            ticker="GBP.TEST",
            currency="GBP",
            current_weight=0.20,
        ),
        "gbx_asset": AssetConfig(
            name="GBX Asset",
            ticker="GBX.TEST",
            currency="GBX",
            current_weight=0.20,
        ),
        "eur_asset": AssetConfig(
            name="EUR Asset",
            ticker="EUR.TEST",
            currency="EUR",
            current_weight=0.20,
        ),
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
            current_weight=0.20,
        ),
        "cad_asset": AssetConfig(
            name="CAD Asset",
            ticker="CAD.TEST",
            currency="CAD",
            current_weight=0.20,
        ),
    }


@pytest.fixture
def dates() -> pd.DatetimeIndex:
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


def test_fx_rates_are_forward_filled(
    dates: pd.DatetimeIndex,
) -> None:
    assets = {
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
            current_weight=1.0,
        )
    }

    prices = pd.DataFrame(
        {
            "USD Asset": [130.0, 143.0],
        },
        index=dates,
    )

    fx_rates = pd.DataFrame(
        {
            "USD": [1.30],
        },
        index=dates[:1],
    )

    converted = convert_prices_to_gbp(
        prices=prices,
        assets=assets,
        fx_rates=fx_rates,
    )

    expected = np.array(
        [
            100.0,
            110.0,
        ]
    )

    np.testing.assert_allclose(
        converted["USD Asset"].to_numpy(),
        expected,
    )


def test_missing_fx_rate_raises_error(
    dates: pd.DatetimeIndex,
) -> None:
    assets = {
        "usd_asset": AssetConfig(
            name="USD Asset",
            ticker="USD.TEST",
            currency="USD",
            current_weight=1.0,
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


def test_unsupported_currency_raises_error(
    dates: pd.DatetimeIndex,
) -> None:
    assets = {
        "jpy_asset": AssetConfig(
            name="JPY Asset",
            ticker="JPY.TEST",
            currency="JPY",
            current_weight=1.0,
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
