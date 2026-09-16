"""Test end-to-end portfolio analysis orchestration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import big_portfolio.pipeline as pipeline_module
from big_portfolio.optimisation.mean_variance import EfficientFrontierResult
from big_portfolio.pipeline import (
    _apply_price_proxies,
    _build_output_paths,
    run_portfolio_analysis,
)


@pytest.fixture
def pipeline_config() -> SimpleNamespace:
    """Return a minimal configuration for pipeline orchestration tests."""
    return SimpleNamespace(
        estimation=SimpleNamespace(
            start_date="2024-01-01",
            end_date="2024-12-31",
            frequency="W-FRI",
            periods_per_year=52,
        ),
        cash=SimpleNamespace(
            expected_return=0.01,
        ),
        assets={
            "asset_a": SimpleNamespace(
                name="Asset A",
                ticker="AAA",
                currency="GBP",
            ),
            "asset_b": SimpleNamespace(
                name="Asset B",
                ticker="BBB",
                currency="USD",
            ),
            "asset_c": SimpleNamespace(
                name="Asset C",
                ticker="CCC",
                currency="EUR",
            ),
        },
        proxies={},
    )


@pytest.fixture
def moment_estimate() -> SimpleNamespace:
    """Return annual moment estimates for three assets."""
    asset_names = [
        "Asset A",
        "Asset B",
        "Asset C",
    ]

    covariance = pd.DataFrame(
        [
            [0.040, 0.010, 0.005],
            [0.010, 0.090, 0.008],
            [0.005, 0.008, 0.025],
        ],
        index=asset_names,
        columns=asset_names,
    )

    return SimpleNamespace(
        expected_returns=pd.Series(
            {
                "Asset A": 0.10,
                "Asset B": 0.15,
                "Asset C": 0.08,
            },
            name="annual_expected_return",
        ),
        covariance=covariance,
        volatilities=pd.Series(
            np.sqrt(np.diag(covariance.to_numpy())),
            index=asset_names,
            name="annual_volatility",
        ),
    )


def test_pipeline_uses_selected_portfolio_assets_and_writes_outputs(
    pipeline_config: SimpleNamespace,
    moment_estimate: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The selected snapshot determines optimisation inputs and its output filenames
    config_path = tmp_path / "portfolio.yaml"
    output_directory = tmp_path / "outputs"

    dates = pd.to_datetime(
        [
            "2024-01-05",
            "2024-01-12",
        ]
    )

    market_prices = pd.DataFrame(
        {
            "Asset A": [
                100.0,
                101.0,
            ],
            "Asset B": [
                80.0,
                82.0,
            ],
            "Asset C": [
                60.0,
                61.0,
            ],
        },
        index=dates,
    )

    fx_rates = pd.DataFrame(
        {
            "EUR": [
                0.86,
                0.85,
            ],
            "USD": [
                1.27,
                1.28,
            ],
        },
        index=dates,
    )

    log_returns = pd.DataFrame(
        {
            "Asset A": [
                0.01,
                0.02,
            ],
            "Asset B": [
                0.02,
                -0.01,
            ],
            "Asset C": [
                0.00,
                0.01,
            ],
        },
        index=dates,
    )

    portfolio_snapshot = SimpleNamespace(
        cash_weight=0.10,
    )
    portfolio_weights = {
        "Asset A": 0.60,
        "Asset B": 0.30,
    }

    diagnostic = object()

    frontier = EfficientFrontierResult(
        summary=pd.DataFrame(
            {
                "risk_aversion": [
                    1.0,
                    0.5,
                ],
                "expected_return": [
                    0.10,
                    0.12,
                ],
                "volatility": [
                    0.12,
                    0.16,
                ],
                "cash_weight": [
                    0.20,
                    0.10,
                ],
            }
        ),
        weights=pd.DataFrame(
            {
                "Asset A": [
                    0.50,
                    0.55,
                ],
                "Asset B": [
                    0.30,
                    0.35,
                ],
                "Cash": [
                    0.20,
                    0.10,
                ],
            }
        ),
    )

    allocation_comparison = pd.DataFrame(
        {
            "current": [
                0.60,
                0.30,
                0.10,
            ],
            "same_risk": [
                0.50,
                0.40,
                0.10,
            ],
            "same_return": [
                0.55,
                0.25,
                0.20,
            ],
        },
        index=[
            "Asset A",
            "Asset B",
            "Cash",
        ],
    )

    load_config = Mock(return_value=pipeline_config)
    download_prices = Mock(return_value=market_prices)
    download_fx = Mock(return_value=fx_rates)
    convert_prices = Mock(return_value=market_prices)
    build_returns = Mock(
        return_value=SimpleNamespace(
            log_returns=log_returns,
        )
    )
    estimate_moments = Mock(return_value=moment_estimate)
    get_snapshot = Mock(return_value=portfolio_snapshot)
    get_weights = Mock(return_value=portfolio_weights)
    analyse_efficiency = Mock(return_value=diagnostic)
    generate_frontier = Mock(return_value=frontier)
    build_comparison = Mock(return_value=allocation_comparison)

    figure = object()

    def fake_plot(
        *,
        frontier: EfficientFrontierResult,
        diagnostic: object,
        output_path: str | Path,
    ) -> object:
        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_bytes(b"figure")

        return figure

    plot_frontier = Mock(side_effect=fake_plot)
    close_figure = Mock()

    monkeypatch.setattr(
        pipeline_module,
        "load_portfolio_config",
        load_config,
    )
    monkeypatch.setattr(
        pipeline_module,
        "download_adjusted_prices",
        download_prices,
    )
    monkeypatch.setattr(
        pipeline_module,
        "download_fx_rates",
        download_fx,
    )
    monkeypatch.setattr(
        pipeline_module,
        "convert_prices_to_gbp",
        convert_prices,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_weekly_log_returns",
        build_returns,
    )
    monkeypatch.setattr(
        pipeline_module,
        "estimate_annual_moments",
        estimate_moments,
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_portfolio_snapshot",
        get_snapshot,
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_portfolio_weights",
        get_weights,
    )
    monkeypatch.setattr(
        pipeline_module,
        "analyse_portfolio_efficiency",
        analyse_efficiency,
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_efficient_frontier",
        generate_frontier,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_allocation_comparison",
        build_comparison,
    )
    monkeypatch.setattr(
        pipeline_module,
        "plot_efficient_frontier",
        plot_frontier,
    )
    monkeypatch.setattr(
        pipeline_module.plt,
        "close",
        close_figure,
    )

    result = run_portfolio_analysis(
        config_path=config_path,
        output_directory=output_directory,
        portfolio_id="current",
    )

    assert result.portfolio_id == "current"
    assert result.asset_statistics.index.tolist() == [
        "Asset A",
        "Asset B",
    ]
    assert result.asset_statistics.columns.tolist() == [
        "annual_expected_return",
        "annual_volatility",
    ]

    analyse_call = analyse_efficiency.call_args.kwargs

    assert analyse_call["expected_returns"].index.tolist() == [
        "Asset A",
        "Asset B",
    ]
    assert analyse_call["covariance"].index.tolist() == [
        "Asset A",
        "Asset B",
    ]
    assert analyse_call["current_weights"].name == "portfolio_weight"
    assert analyse_call["current_cash_weight"] == pytest.approx(0.10)

    frontier_call = generate_frontier.call_args.kwargs

    assert frontier_call["risk_aversions"] == pipeline_module.DEFAULT_RISK_AVERSIONS

    download_fx.assert_called_once_with(
        currencies={
            "GBP",
            "USD",
            "EUR",
        },
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    expected_output_names = {
        "asset_statistics_current.csv",
        "covariance_current.csv",
        "allocation_comparison_current.csv",
        "frontier_summary_current.csv",
        "frontier_weights_current.csv",
    }

    written_table_names = {
        path.name for path in (output_directory / "tables").iterdir()
    }

    assert written_table_names == expected_output_names
    assert (output_directory / "figures" / "efficient_frontier_current.png").exists()

    close_figure.assert_called_once_with(figure)


def test_price_proxies_are_applied_in_configuration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each enabled proxy receives the prices produced by the previous backfill
    prices = pd.DataFrame(
        {
            "Proxy One": [
                np.nan,
                100.0,
            ],
            "Proxy Two": [
                np.nan,
                200.0,
            ],
        },
        index=pd.to_datetime(
            [
                "2024-01-05",
                "2024-01-12",
            ]
        ),
    )

    first_stage = prices.copy()
    second_stage = prices.copy()

    config = SimpleNamespace(
        estimation=SimpleNamespace(
            start_date="2024-01-01",
            end_date="2024-12-31",
        ),
        assets={
            "proxy_one": SimpleNamespace(
                name="Proxy One",
            ),
            "proxy_two": SimpleNamespace(
                name="Proxy Two",
            ),
            "disabled_proxy": SimpleNamespace(
                name="Disabled Proxy",
            ),
        },
        proxies={
            "proxy_one": SimpleNamespace(
                provider=pipeline_module.BANK_OF_ENGLAND_PROVIDER,
                series_code="SERIES_ONE",
                apply_before_first_observation=True,
            ),
            "proxy_two": SimpleNamespace(
                provider=pipeline_module.BANK_OF_ENGLAND_PROVIDER,
                series_code="SERIES_TWO",
                apply_before_first_observation=True,
            ),
            "disabled_proxy": SimpleNamespace(
                provider="Unsupported Provider",
                series_code="DISABLED",
                apply_before_first_observation=False,
            ),
        },
    )

    first_proxy_index = pd.Series(
        [
            100.0,
            101.0,
        ],
        index=prices.index,
    )
    second_proxy_index = pd.Series(
        [
            200.0,
            201.0,
        ],
        index=prices.index,
    )

    download_proxy = Mock(
        side_effect=[
            first_proxy_index,
            second_proxy_index,
        ]
    )
    backfill_proxy = Mock(
        side_effect=[
            SimpleNamespace(
                prices=first_stage,
            ),
            SimpleNamespace(
                prices=second_stage,
            ),
        ]
    )

    monkeypatch.setattr(
        pipeline_module,
        "download_boe_series",
        download_proxy,
    )
    monkeypatch.setattr(
        pipeline_module,
        "backfill_price_with_proxy",
        backfill_proxy,
    )

    result = _apply_price_proxies(
        prices=prices,
        config=config,
    )

    assert result is second_stage
    assert download_proxy.call_count == 2
    assert backfill_proxy.call_count == 2

    assert download_proxy.call_args_list[0].kwargs["series_code"] == "SERIES_ONE"
    assert download_proxy.call_args_list[1].kwargs["series_code"] == "SERIES_TWO"

    assert backfill_proxy.call_args_list[0].kwargs["asset_name"] == "Proxy One"
    assert backfill_proxy.call_args_list[1].kwargs["asset_name"] == "Proxy Two"
    assert backfill_proxy.call_args_list[1].kwargs["prices"] is first_stage


def test_unsupported_enabled_proxy_provider_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An enabled proxy must use a provider implemented by the pipeline
    config = SimpleNamespace(
        estimation=SimpleNamespace(
            start_date="2024-01-01",
            end_date="2024-12-31",
        ),
        assets={
            "asset_a": SimpleNamespace(
                name="Asset A",
            ),
        },
        proxies={
            "asset_a": SimpleNamespace(
                provider="Unsupported Provider",
                series_code="TEST_SERIES",
                apply_before_first_observation=True,
            ),
        },
    )

    download_proxy = Mock()

    monkeypatch.setattr(
        pipeline_module,
        "download_boe_series",
        download_proxy,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported proxy provider",
    ):
        _apply_price_proxies(
            prices=pd.DataFrame(
                {
                    "Asset A": [
                        100.0,
                    ]
                }
            ),
            config=config,
        )

    download_proxy.assert_not_called()


def test_output_paths_are_distinct_between_portfolios(
    tmp_path: Path,
) -> None:
    # Different portfolio snapshots must resolve to different output files
    current_paths = _build_output_paths(
        output_directory=tmp_path,
        portfolio_id="current",
    )
    historical_paths = _build_output_paths(
        output_directory=tmp_path,
        portfolio_id="pre_diversification",
    )

    current_values = set(current_paths.__dict__.values())
    historical_values = set(historical_paths.__dict__.values())

    assert current_values.isdisjoint(historical_values)

    assert current_paths.asset_statistics.name == ("asset_statistics_current.csv")
    assert historical_paths.asset_statistics.name == (
        "asset_statistics_pre_diversification.csv"
    )

    assert current_paths.efficient_frontier.name == ("efficient_frontier_current.png")
    assert historical_paths.efficient_frontier.name == (
        "efficient_frontier_pre_diversification.png"
    )
