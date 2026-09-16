"""Plot portfolio sensitivity to correlation stress scenarios."""

from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
from pandas.api.types import is_numeric_dtype

from big_portfolio.analysis.correlation_stress import CorrelationStressResult

REQUIRED_SUMMARY_COLUMNS = {
    "scenario",
    "current_volatility",
    "variance_multiplier",
}


def plot_variance_stress_comparison(
    results: dict[str, CorrelationStressResult],
    output_path: str | Path | None = None,
) -> Figure:
    """Compare portfolio variance sensitivity across stress scenarios."""
    _validate_results(results)

    figure, axis = plt.subplots(figsize=(9, 6))

    for portfolio_name, result in results.items():
        axis.plot(
            result.summary["scenario"],
            result.summary["variance_multiplier"],
            marker="o",
            linewidth=2.0,
            label=portfolio_name,
        )

    # A multiplier of one represents the unstressed portfolio variance
    axis.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
    )

    axis.set_title("Portfolio Variance Under Correlation Stress")
    axis.set_xlabel("Within-group correlation floor")
    axis.set_ylabel("Portfolio variance multiplier")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    if output_path is not None:
        _save_figure(
            figure=figure,
            output_path=output_path,
        )

    return figure


def plot_volatility_stress_comparison(
    results: dict[str, CorrelationStressResult],
    output_path: str | Path | None = None,
) -> Figure:
    """Compare portfolio volatility across correlation stress scenarios."""
    _validate_results(results)

    figure, axis = plt.subplots(figsize=(9, 6))

    for portfolio_name, result in results.items():
        axis.plot(
            result.summary["scenario"],
            result.summary["current_volatility"],
            marker="o",
            linewidth=2.0,
            label=portfolio_name,
        )

    axis.set_title("Portfolio Volatility Under Correlation Stress")
    axis.set_xlabel("Within-group correlation floor")
    axis.set_ylabel("Annual volatility")
    axis.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    if output_path is not None:
        _save_figure(
            figure=figure,
            output_path=output_path,
        )

    return figure


def plot_allocation_shift(
    result: CorrelationStressResult,
    stress_scenario: str,
    output_path: str | Path | None = None,
    minimum_change: float = 0.0025,
) -> Figure:
    """Plot efficient allocation changes from the base to a stress scenario."""
    _validate_minimum_change(minimum_change)

    weights = result.efficient_weights
    _validate_weight_table(weights)

    if "Base" not in weights.index:
        raise ValueError("Efficient weights must contain a Base scenario.")

    if stress_scenario not in weights.index:
        raise ValueError(f"Unknown stress scenario: '{stress_scenario}'.")

    if stress_scenario == "Base":
        raise ValueError("stress_scenario must identify a non-base scenario.")

    weight_change = weights.loc[stress_scenario] - weights.loc["Base"]

    # Apply the minimum-change threshold to absolute weight changes
    weight_change = weight_change[weight_change.abs() >= minimum_change].sort_values()

    if weight_change.empty:
        raise ValueError("No allocation changes meet the minimum threshold.")

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.barh(
        weight_change.index,
        weight_change.to_numpy(),
    )

    axis.axvline(
        0.0,
        linewidth=1.0,
    )

    axis.set_title(f"Same-Return Allocation Shift: {stress_scenario} Correlation Floor")
    axis.set_xlabel("Change in portfolio weight")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.1%}"))
    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()

    if output_path is not None:
        _save_figure(
            figure=figure,
            output_path=output_path,
        )

    return figure


def _validate_results(
    results: dict[str, CorrelationStressResult],
) -> None:
    """Validate correlation-stress results before plotting."""
    if not results:
        raise ValueError("At least one stress result is required.")

    for portfolio_name, result in results.items():
        summary = result.summary

        if summary.empty:
            raise ValueError(f"Stress result '{portfolio_name}' has an empty summary.")

        missing_columns = REQUIRED_SUMMARY_COLUMNS - set(summary.columns)

        if missing_columns:
            raise ValueError(
                f"Stress result '{portfolio_name}' is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if summary["scenario"].isna().any():
            raise ValueError(
                f"Stress result '{portfolio_name}' contains missing scenario labels."
            )

        if not summary["scenario"].is_unique:
            raise ValueError(
                f"Stress result '{portfolio_name}' contains duplicate scenario labels."
            )

        for column in (
            "current_volatility",
            "variance_multiplier",
        ):
            if not is_numeric_dtype(summary[column].dtype):
                raise TypeError(
                    f"Stress result '{portfolio_name}' column "
                    f"'{column}' must be numeric."
                )

            values = summary[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Stress result '{portfolio_name}' column "
                    f"'{column}' must contain only finite values."
                )


def _validate_weight_table(
    weights: pd.DataFrame,
) -> None:
    """Validate efficient portfolio weights before plotting changes."""
    if weights.empty:
        raise ValueError("Efficient weight table cannot be empty.")

    if not weights.index.is_unique:
        raise ValueError("Efficient weight scenario labels must be unique.")

    if not weights.columns.is_unique:
        raise ValueError("Efficient weight asset names must be unique.")

    non_numeric_columns = [
        column
        for column in weights.columns
        if not is_numeric_dtype(weights[column].dtype)
    ]

    if non_numeric_columns:
        raise TypeError(
            f"Efficient weights must be numeric for: {sorted(non_numeric_columns)}"
        )

    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Efficient weights must contain only finite values.")


def _validate_minimum_change(
    minimum_change: float,
) -> None:
    """Validate the allocation-change display threshold."""
    if not isfinite(minimum_change):
        raise ValueError("minimum_change must be finite.")

    if minimum_change < 0.0:
        raise ValueError("minimum_change cannot be negative.")


def _save_figure(
    figure: Figure,
    output_path: str | Path,
) -> None:
    """Save a high-resolution figure to disk."""
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
