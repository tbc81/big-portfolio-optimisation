"""Test efficient-frontier visualisation and figure output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from big_portfolio.analysis.diagnostics import (
    PortfolioDiagnosticResult,
    PortfolioSnapshot,
)
from big_portfolio.optimisation.mean_variance import (
    EfficientFrontierResult,
    PortfolioSolution,
)
from big_portfolio.visualisation.frontier import plot_efficient_frontier


@pytest.fixture
def frontier() -> EfficientFrontierResult:
    """Return representative efficient-frontier plotting data."""
    summary = pd.DataFrame(
        {
            "risk_aversion": [
                0.1,
                1.0,
                10.0,
            ],
            "expected_return": [
                0.20,
                0.15,
                0.10,
            ],
            "volatility": [
                0.25,
                0.18,
                0.12,
            ],
            "cash_weight": [
                0.0,
                0.1,
                0.3,
            ],
        }
    )

    weights = pd.DataFrame(
        {
            "Asset A": [
                0.30,
                0.45,
                0.40,
            ],
            "Asset B": [
                0.70,
                0.45,
                0.30,
            ],
            "Cash": [
                0.0,
                0.10,
                0.30,
            ],
        }
    )

    return EfficientFrontierResult(
        summary=summary,
        weights=weights,
    )


@pytest.fixture
def diagnostic() -> PortfolioDiagnosticResult:
    """Return representative portfolio efficiency diagnostics."""
    current = PortfolioSnapshot(
        weights=pd.Series(
            {
                "Asset A": 0.50,
                "Asset B": 0.40,
            }
        ),
        cash_weight=0.10,
        expected_return=0.13,
        volatility=0.20,
    )

    same_risk = PortfolioSolution(
        weights=pd.Series(
            {
                "Asset A": 0.40,
                "Asset B": 0.60,
            }
        ),
        cash_weight=0.0,
        expected_return=0.17,
        volatility=0.20,
    )

    same_return = PortfolioSolution(
        weights=pd.Series(
            {
                "Asset A": 0.45,
                "Asset B": 0.35,
            }
        ),
        cash_weight=0.20,
        expected_return=0.13,
        volatility=0.14,
    )

    return PortfolioDiagnosticResult(
        current=current,
        same_risk=same_risk,
        same_return=same_return,
        return_improvement=0.04,
        volatility_reduction=0.06,
    )


def test_frontier_plot_uses_sorted_risk_return_data(
    frontier: EfficientFrontierResult,
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    # The frontier line should progress from lower to higher portfolio volatility
    figure = plot_efficient_frontier(
        frontier=frontier,
        diagnostic=diagnostic,
    )

    axis = figure.axes[0]
    frontier_line = axis.lines[0]

    assert isinstance(figure, Figure)
    assert axis.get_title() == "Portfolio Efficient Frontier"
    assert axis.get_xlabel() == "Annual volatility"
    assert axis.get_ylabel() == "Annual expected return"

    np.testing.assert_allclose(
        frontier_line.get_xdata(),
        np.array(
            [
                0.12,
                0.18,
                0.25,
            ]
        ),
    )
    np.testing.assert_allclose(
        frontier_line.get_ydata(),
        np.array(
            [
                0.10,
                0.15,
                0.20,
            ]
        ),
    )

    plt.close(figure)


def test_frontier_plot_contains_diagnostic_points(
    frontier: EfficientFrontierResult,
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    # Scatter points should represent the portfolio and both efficient comparisons
    figure = plot_efficient_frontier(
        frontier=frontier,
        diagnostic=diagnostic,
    )

    axis = figure.axes[0]

    plotted_points = np.vstack(
        [collection.get_offsets()[0] for collection in axis.collections]
    )

    expected_points = np.array(
        [
            [0.20, 0.13],
            [0.20, 0.17],
            [0.14, 0.13],
        ]
    )

    np.testing.assert_allclose(
        plotted_points,
        expected_points,
    )

    plt.close(figure)


def test_frontier_plot_contains_expected_labels(
    frontier: EfficientFrontierResult,
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    # The legend identifies the frontier and three diagnostic portfolios
    figure = plot_efficient_frontier(
        frontier=frontier,
        diagnostic=diagnostic,
    )

    axis = figure.axes[0]
    legend = axis.get_legend()

    assert legend is not None

    legend_labels = [text.get_text() for text in legend.get_texts()]

    assert legend_labels == [
        "Efficient frontier",
        "Portfolio",
        "Same-risk efficient",
        "Same-return efficient",
    ]

    plt.close(figure)


def test_frontier_figure_is_saved(
    frontier: EfficientFrontierResult,
    diagnostic: PortfolioDiagnosticResult,
    tmp_path: Path,
) -> None:
    # Saving should create missing directories and write a non-empty image file
    output_path = tmp_path / "figures" / "frontier" / "efficient_frontier.png"

    figure = plot_efficient_frontier(
        frontier=frontier,
        diagnostic=diagnostic,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_empty_frontier_raises_error(
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    # An efficient-frontier chart requires at least one risk-return observation
    frontier = EfficientFrontierResult(
        summary=pd.DataFrame(),
        weights=pd.DataFrame(),
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        plot_efficient_frontier(
            frontier=frontier,
            diagnostic=diagnostic,
        )


def test_missing_frontier_column_raises_error(
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    # Frontier plotting requires both expected-return and volatility statistics
    frontier = EfficientFrontierResult(
        summary=pd.DataFrame(
            {
                "expected_return": [
                    0.10,
                    0.15,
                ]
            }
        ),
        weights=pd.DataFrame(),
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        plot_efficient_frontier(
            frontier=frontier,
            diagnostic=diagnostic,
        )


def test_non_numeric_frontier_column_raises_error(
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    # Frontier statistics should reach the visualisation layer as numerical data
    frontier = EfficientFrontierResult(
        summary=pd.DataFrame(
            {
                "expected_return": [
                    0.10,
                    0.15,
                ],
                "volatility": [
                    "0.12",
                    "0.18",
                ],
            }
        ),
        weights=pd.DataFrame(),
    )

    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        plot_efficient_frontier(
            frontier=frontier,
            diagnostic=diagnostic,
        )


def test_non_finite_frontier_value_raises_error(
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    # Infinite frontier statistics cannot represent valid chart coordinates
    frontier = EfficientFrontierResult(
        summary=pd.DataFrame(
            {
                "expected_return": [
                    0.10,
                    np.inf,
                ],
                "volatility": [
                    0.12,
                    0.18,
                ],
            }
        ),
        weights=pd.DataFrame(),
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        plot_efficient_frontier(
            frontier=frontier,
            diagnostic=diagnostic,
        )


def test_negative_frontier_volatility_raises_error(
    diagnostic: PortfolioDiagnosticResult,
) -> None:
    # Portfolio volatility cannot be negative on a valid efficient frontier
    frontier = EfficientFrontierResult(
        summary=pd.DataFrame(
            {
                "expected_return": [
                    0.10,
                    0.15,
                ],
                "volatility": [
                    -0.01,
                    0.18,
                ],
            }
        ),
        weights=pd.DataFrame(),
    )

    with pytest.raises(
        ValueError,
        match="volatility cannot be negative",
    ):
        plot_efficient_frontier(
            frontier=frontier,
            diagnostic=diagnostic,
        )
