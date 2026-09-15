import pandas as pd

from big_portfolio.config import AssetConfig

SUPPORTED_CURRENCIES = {
    "GBP",
    "GBX",
    "EUR",
    "USD",
    "CAD",
}


def convert_prices_to_gbp(
    prices: pd.DataFrame,
    assets: dict[str, AssetConfig],
    fx_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Convert portfolio prices from local currencies into GBP."""
    if prices.empty:
        raise ValueError("Price data cannot be empty.")

    asset_names = [asset.name for asset in assets.values()]

    # Validates that all expected assets have been downloaded
    missing_assets = set(asset_names) - set(prices.columns)

    if missing_assets:
        raise ValueError(f"Missing price data for: {sorted(missing_assets)}")

    # Validates that all supported FX rates have been downloaded
    currencies = {asset.currency for asset in assets.values()}

    unsupported_currencies = currencies - SUPPORTED_CURRENCIES

    if unsupported_currencies:
        raise ValueError(f"Unsupported currencies: {sorted(unsupported_currencies)}")

    # Validates that all required FX rates have been downloaded
    required_fx = sorted(currencies - {"GBP", "GBX"})

    missing_fx = set(required_fx) - set(fx_rates.columns)

    if missing_fx:
        raise ValueError(f"Missing FX rates for: {sorted(missing_fx)}")

    # Aligns FX rates with price data
    aligned_fx = fx_rates.reindex(prices.index).ffill()

    if required_fx and aligned_fx[required_fx].isna().any().any():
        raise ValueError("FX data contains missing values after alignment.")

    gbp_prices = pd.DataFrame(
        index=prices.index,
        columns=asset_names,
        dtype=float,
    )

    # Converts local prices to GBP based on asset currency and FX rates
    for asset in assets.values():
        local_prices = prices[asset.name]

        if asset.currency == "GBP":
            gbp_prices[asset.name] = local_prices

        elif asset.currency == "GBX":
            gbp_prices[asset.name] = local_prices / 100

        elif asset.currency == "EUR":
            gbp_prices[asset.name] = local_prices * aligned_fx["EUR"]

        elif asset.currency == "USD":
            gbp_prices[asset.name] = local_prices / aligned_fx["USD"]

        elif asset.currency == "CAD":
            gbp_prices[asset.name] = local_prices / aligned_fx["CAD"]

    return gbp_prices
