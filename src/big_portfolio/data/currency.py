"""Convert local-currency asset prices into GBP."""

import numpy as np
import pandas as pd

from big_portfolio.config import AssetConfig

FX_CURRENCIES = frozenset({"EUR", "USD", "CAD"})
SUPPORTED_CURRENCIES = frozenset({"GBP", "GBX"}) | FX_CURRENCIES


def convert_prices_to_gbp(
    prices: pd.DataFrame,
    assets: dict[str, AssetConfig],
    fx_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Convert configured asset prices from local currencies into GBP."""
    if prices.empty:
        raise ValueError("Price data cannot be empty.")

    if not assets:
        raise ValueError("At least one asset is required for currency conversion.")

    _validate_time_index(
        index=prices.index,
        name="Price",
    )

    asset_names = [asset.name for asset in assets.values()]
    missing_assets = set(asset_names) - set(prices.columns)

    if missing_assets:
        raise ValueError(f"Missing price data for: {sorted(missing_assets)}")

    currencies = {asset.currency for asset in assets.values()}
    unsupported_currencies = currencies - SUPPORTED_CURRENCIES

    if unsupported_currencies:
        raise ValueError(f"Unsupported currencies: {sorted(unsupported_currencies)}")

    required_fx = sorted(currencies & FX_CURRENCIES)
    missing_fx = set(required_fx) - set(fx_rates.columns)

    if missing_fx:
        raise ValueError(f"Missing FX rates for: {sorted(missing_fx)}")

    if required_fx:
        _validate_time_index(
            index=fx_rates.index,
            name="FX",
        )

        # Use the latest FX observation available on or before each asset-price date
        aligned_fx = fx_rates.reindex(
            prices.index,
            method="ffill",
        )[required_fx].apply(pd.to_numeric, errors="raise")

        fx_values = aligned_fx.to_numpy(dtype=float)

        if not np.isfinite(fx_values).all():
            raise ValueError(
                "FX data contains missing or non-finite values after alignment."
            )

        if (fx_values <= 0.0).any():
            raise ValueError("FX rates must be positive.")
    else:
        aligned_fx = pd.DataFrame(index=prices.index)

    gbp_prices = pd.DataFrame(
        index=prices.index,
        columns=asset_names,
        dtype=float,
    )

    for asset in assets.values():
        local_prices = prices[asset.name]

        match asset.currency:
            case "GBP":
                gbp_prices[asset.name] = local_prices

            case "GBX":
                # London-listed GBX prices are quoted in pence, so divide by 100
                gbp_prices[asset.name] = local_prices / 100.0

            case "EUR":
                # EURGBP quotes GBP per EUR: GBP price = EUR price × EURGBP
                gbp_prices[asset.name] = local_prices * aligned_fx["EUR"]

            case "USD" | "CAD":
                # GBPUSD and GBPCAD quote foreign currency per GBP, so divide by FX
                gbp_prices[asset.name] = local_prices / aligned_fx[asset.currency]

    return gbp_prices


def _validate_time_index(
    index: pd.Index,
    name: str,
) -> None:
    """Validate a time-series index used for historical alignment."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} data must use a DatetimeIndex.")

    if index.has_duplicates:
        raise ValueError(f"{name} data contains duplicate dates.")

    # Forward FX alignment requires observations to be ordered through time
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} data must be sorted by date.")
