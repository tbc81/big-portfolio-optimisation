"""Provide the command-line interface for portfolio analysis."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from big_portfolio.pipeline import PipelineResult, run_portfolio_analysis


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the portfolio analysis command-line interface."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "run":
        result = run_portfolio_analysis(
            config_path=arguments.config,
            output_directory=arguments.output_dir,
            portfolio_id=arguments.portfolio_id,
        )

        _print_run_summary(result)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="portfolio-analytics",
        description="Run portfolio optimisation and risk analysis.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a complete portfolio analysis.",
    )

    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/portfolio.yaml"),
        help="Path to the portfolio configuration file.",
    )

    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated tables and figures.",
    )

    run_parser.add_argument(
        "--portfolio-id",
        default="current",
        help="Configured portfolio snapshot to analyse.",
    )

    return parser


def _print_run_summary(
    result: PipelineResult,
) -> None:
    """Print the principal portfolio diagnostic results."""
    diagnostic = result.diagnostic

    print(f"\nPortfolio Analysis: {result.portfolio_id}")
    print("-" * (20 + len(result.portfolio_id)))

    print(f"Portfolio expected return: {diagnostic.current.expected_return:.2%}")
    print(f"Portfolio volatility:      {diagnostic.current.volatility:.2%}")

    print("\nSame-risk efficient portfolio")
    print(f"Expected return:            {diagnostic.same_risk.expected_return:.2%}")
    print(f"Volatility:                 {diagnostic.same_risk.volatility:.2%}")
    print(f"Return improvement:         {diagnostic.return_improvement:.2%}")

    print("\nSame-return efficient portfolio")
    print(f"Expected return:            {diagnostic.same_return.expected_return:.2%}")
    print(f"Volatility:                 {diagnostic.same_return.volatility:.2%}")
    print(f"Volatility reduction:       {diagnostic.volatility_reduction:.2%}")

    print("\nOutputs")
    print(f"Efficient frontier:         {result.outputs.efficient_frontier}")
    print(f"Allocation comparison:      {result.outputs.allocation_comparison}")
