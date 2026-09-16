"""Test command-line parsing, execution and summary output."""

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import big_portfolio.cli as cli_module
from big_portfolio.analysis.diagnostics import (
    PortfolioDiagnosticResult,
    PortfolioSnapshot,
)
from big_portfolio.cli import (
    _build_parser,
    _print_run_summary,
    main,
)
from big_portfolio.optimisation.mean_variance import (
    EfficientFrontierResult,
    PortfolioSolution,
)
from big_portfolio.pipeline import (
    OutputPaths,
    PipelineResult,
)


@pytest.fixture
def pipeline_result(
    tmp_path: Path,
) -> PipelineResult:
    """Return representative completed pipeline results for CLI tests."""
    current = PortfolioSnapshot(
        weights=pd.Series(
            {
                "Asset A": 0.90,
            }
        ),
        cash_weight=0.10,
        expected_return=0.10,
        volatility=0.15,
    )

    same_risk = PortfolioSolution(
        weights=pd.Series(
            {
                "Asset A": 1.0,
            }
        ),
        cash_weight=0.0,
        expected_return=0.14,
        volatility=0.15,
    )

    same_return = PortfolioSolution(
        weights=pd.Series(
            {
                "Asset A": 0.70,
            }
        ),
        cash_weight=0.30,
        expected_return=0.10,
        volatility=0.09,
    )

    diagnostic = PortfolioDiagnosticResult(
        current=current,
        same_risk=same_risk,
        same_return=same_return,
        return_improvement=0.04,
        volatility_reduction=0.06,
    )

    outputs = OutputPaths(
        asset_statistics=(tmp_path / "asset_statistics_current.csv"),
        covariance=(tmp_path / "covariance_current.csv"),
        allocation_comparison=(tmp_path / "allocation_comparison_current.csv"),
        frontier_summary=(tmp_path / "frontier_summary_current.csv"),
        frontier_weights=(tmp_path / "frontier_weights_current.csv"),
        efficient_frontier=(tmp_path / "efficient_frontier_current.png"),
    )

    return PipelineResult(
        portfolio_id="current",
        diagnostic=diagnostic,
        frontier=EfficientFrontierResult(
            summary=pd.DataFrame(),
            weights=pd.DataFrame(),
        ),
        asset_statistics=pd.DataFrame(),
        allocation_comparison=pd.DataFrame(),
        outputs=outputs,
    )


def test_parser_uses_public_command_name() -> None:
    # Parser usage should match the executable defined in pyproject.toml
    parser = _build_parser()

    assert parser.prog == "portfolio-analytics"


def test_run_parser_uses_default_arguments() -> None:
    # The run command should use the standard config, output and portfolio defaults
    parser = _build_parser()

    arguments = parser.parse_args(
        [
            "run",
        ]
    )

    assert arguments.command == "run"
    assert arguments.config == Path("config/portfolio.yaml")
    assert arguments.output_dir == Path("outputs")
    assert arguments.portfolio_id == "current"


def test_run_parser_accepts_custom_arguments() -> None:
    # User-supplied paths and portfolio identifiers should override all defaults
    parser = _build_parser()

    arguments = parser.parse_args(
        [
            "run",
            "--config",
            "custom.yaml",
            "--output-dir",
            "results",
            "--portfolio-id",
            "pre_diversification",
        ]
    )

    assert arguments.config == Path("custom.yaml")
    assert arguments.output_dir == Path("results")
    assert arguments.portfolio_id == ("pre_diversification")


def test_main_runs_selected_portfolio(
    pipeline_result: PipelineResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Parsed run arguments should be forwarded to the portfolio pipeline
    run_analysis = Mock(return_value=pipeline_result)
    print_summary = Mock()

    monkeypatch.setattr(
        cli_module,
        "run_portfolio_analysis",
        run_analysis,
    )
    monkeypatch.setattr(
        cli_module,
        "_print_run_summary",
        print_summary,
    )

    main(
        [
            "run",
            "--config",
            "custom.yaml",
            "--output-dir",
            "results",
            "--portfolio-id",
            "pre_diversification",
        ]
    )

    run_analysis.assert_called_once_with(
        config_path=Path("custom.yaml"),
        output_directory=Path("results"),
        portfolio_id="pre_diversification",
    )
    print_summary.assert_called_once_with(pipeline_result)


def test_summary_contains_key_metrics(
    pipeline_result: PipelineResult,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Printed output should identify the portfolio and report its core diagnostics
    _print_run_summary(pipeline_result)

    captured = capsys.readouterr()

    assert "Portfolio Analysis: current" in captured.out
    assert "Portfolio expected return: 10.00%" in captured.out
    assert "Portfolio volatility:      15.00%" in captured.out
    assert "Return improvement:         4.00%" in captured.out
    assert "Volatility reduction:       6.00%" in captured.out
    assert "efficient_frontier_current.png" in captured.out
    assert "allocation_comparison_current.csv" in captured.out
