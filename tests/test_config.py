from big_portfolio.config import load_portfolio_config


def test_portfolio_weights_sum_to_one() -> None:
    config = load_portfolio_config("config/portfolio.yaml")

    invested_weight = sum(asset.current_weight for asset in config.assets.values())

    total_weight = invested_weight + config.cash.current_weight

    assert total_weight == 1.0


def test_portfolio_contains_expected_assets() -> None:
    config = load_portfolio_config("config/portfolio.yaml")

    assert len(config.assets) == 18
    assert config.assets["cameco"].ticker == "CCO.TO"
    assert config.assets["cameco"].currency == "CAD"


def test_portfolio_base_currency_is_gbp() -> None:
    config = load_portfolio_config("config/portfolio.yaml")

    assert config.base_currency == "GBP"
