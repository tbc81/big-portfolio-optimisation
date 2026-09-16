"""Plot efficient-frontier and portfolio diagnostic results."""

from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from pandas.api.types import is_numeric_dtype

from big_portfolio.analysis.diagnostics import PortfolioDiagnosticResult
from big_portfolio.optimisation.mean_variance import EfficientFrontierResult

REQUIRED_FRONTIER_COLUMNS = {
    "expected_return",
    "volatility",
}


def plot_efficient_frontier(
    frontier: EfficientFrontierResult,
    diagnostic: PortfolioDiagnosticResult,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot an efficient frontier with portfolio diagnostic comparisons."""
    frontier_summary = _prepare_frontier_summary(frontier.summary)
    _validate_diagnostic(diagnostic)

    figure, axis = plt.subplots(figsize=(9, 6))

    # Volatility ordering produces a continuous risk-return curve
    axis.plot(
        frontier_summary["volatility"],
        frontier_summary["expected_return"],
        linewidth=2.0,
        label="Efficient frontier",
    )

    axis.scatter(
        diagnostic.current.volatility,
        diagnostic.current.expected_return,
        marker="o",
        s=80,
        label="Portfolio",
        zorder=3,
    )

    axis.scatter(
        diagnostic.same_risk.volatility,
        diagnostic.same_risk.expected_return,
        marker="^",
        s=80,
        label="Same-risk efficient",
        zorder=3,
    )

    axis.scatter(
        diagnostic.same_return.volatility,
        diagnostic.same_return.expected_return,
        marker="s",
        s=80,
        label="Same-return efficient",
        zorder=3,
    )

    axis.set_title("Portfolio Efficient Frontier")
    axis.set_xlabel("Annual volatility")
    axis.set_ylabel("Annual expected return")

    percentage_formatter = PercentFormatter(
        xmax=1.0,
        decimals=0,
    )
    axis.xaxis.set_major_formatter(percentage_formatter)
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


def _prepare_frontier_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and order efficient-frontier statistics."""
    if summary.empty:
        raise ValueError("Frontier summary cannot be empty.")

    missing_columns = REQUIRED_FRONTIER_COLUMNS - set(summary.columns)

    if missing_columns:
        raise ValueError(
            f"Frontier summary is missing columns: {sorted(missing_columns)}"
        )

    for column in REQUIRED_FRONTIER_COLUMNS:
        if not is_numeric_dtype(summary[column].dtype):
            raise TypeError(f"Frontier column '{column}' must be numeric.")

    frontier_summary = summary.loc[
        :,
        [
            "expected_return",
            "volatility",
        ],
    ].copy()

    values = frontier_summary.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Frontier summary must contain only finite values.")

    if (frontier_summary["volatility"].to_numpy() < 0.0).any():
        raise ValueError("Frontier volatility cannot be negative.")

    return frontier_summary.sort_values(
        [
            "volatility",
            "expected_return",
        ]
    )


def _validate_diagnostic(
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    """Validate portfolio diagnostic points before plotting."""
    points = {
        "portfolio": (
            diagnostic.current.volatility,
            diagnostic.current.expected_return,
        ),
        "same-risk portfolio": (
            diagnostic.same_risk.volatility,
            diagnostic.same_risk.expected_return,
        ),
        "same-return portfolio": (
            diagnostic.same_return.volatility,
            diagnostic.same_return.expected_return,
        ),
    }

    for point_name, (
        volatility,
        expected_return,
    ) in points.items():
        if not isfinite(volatility) or not isfinite(expected_return):
            raise ValueError(f"{point_name.capitalize()} statistics must be finite.")

        if volatility < 0.0:
            raise ValueError(
                f"{point_name.capitalize()} volatility cannot be negative."
            )


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
