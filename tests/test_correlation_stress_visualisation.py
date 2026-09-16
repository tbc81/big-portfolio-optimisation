"""Test correlation-stress visualisation and figure output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from big_portfolio.analysis.correlation_stress import CorrelationStressResult
from big_portfolio.visualisation.correlation_stress_visualisation import (
    plot_allocation_shift,
    plot_variance_stress_comparison,
    plot_volatility_stress_comparison,
)


@pytest.fixture
def stress_result() -> CorrelationStressResult:
    """Return representative correlation-stress plotting data."""
    summary = pd.DataFrame(
        {
            "scenario": [
                "Base",
                "60%",
                "85%",
            ],
            "target_correlation": [
                None,
                0.60,
                0.85,
            ],
            "realised_group_correlation": [
                0.35,
                0.60,
                0.85,
            ],
            "current_volatility": [
                0.20,
                0.22,
                0.25,
            ],
            "volatility_change": [
                0.00,
                0.02,
                0.05,
            ],
            "variance_multiplier": [
                1.00,
                1.21,
                1.56,
            ],
            "efficient_volatility": [
                0.12,
                0.13,
                0.14,
            ],
            "matrix_distance": [
                0.0,
                1.0,
                2.0,
            ],
        }
    )

    efficient_weights = pd.DataFrame(
        {
            "Asset A": [
                0.50,
                0.45,
                0.35,
            ],
            "Asset B": [
                0.40,
                0.45,
                0.55,
            ],
            "Cash": [
                0.10,
                0.10,
                0.10,
            ],
        },
        index=[
            "Base",
            "60%",
            "85%",
        ],
    )

    return CorrelationStressResult(
        base_correlation=pd.DataFrame(
            [
                [1.0, 0.35],
                [0.35, 1.0],
            ],
            index=[
                "Asset A",
                "Asset B",
            ],
            columns=[
                "Asset A",
                "Asset B",
            ],
        ),
        scenarios=(),
        summary=summary,
        efficient_weights=efficient_weights,
    )


def test_variance_plot_uses_stress_summary(
    stress_result: CorrelationStressResult,
) -> None:
    # The variance chart should plot each scenario's reported variance multiplier
    figure = plot_variance_stress_comparison(
        {
            "Portfolio": stress_result,
        }
    )

    axis = figure.axes[0]
    portfolio_line = axis.lines[0]

    assert isinstance(figure, Figure)
    assert axis.get_title() == "Portfolio Variance Under Correlation Stress"
    assert axis.get_ylabel() == "Portfolio variance multiplier"

    np.testing.assert_allclose(
        portfolio_line.get_ydata(),
        np.array(
            [
                1.00,
                1.21,
                1.56,
            ]
        ),
    )

    plt.close(figure)


def test_volatility_plot_uses_stress_summary(
    stress_result: CorrelationStressResult,
) -> None:
    # The volatility chart should plot the portfolio risk from each stress scenario
    figure = plot_volatility_stress_comparison(
        {
            "Portfolio": stress_result,
        }
    )

    axis = figure.axes[0]
    portfolio_line = axis.lines[0]

    assert isinstance(figure, Figure)
    assert axis.get_title() == "Portfolio Volatility Under Correlation Stress"
    assert axis.get_ylabel() == "Annual volatility"

    np.testing.assert_allclose(
        portfolio_line.get_ydata(),
        np.array(
            [
                0.20,
                0.22,
                0.25,
            ]
        ),
    )

    plt.close(figure)


def test_allocation_shift_plots_weight_changes(
    stress_result: CorrelationStressResult,
) -> None:
    # Bar lengths should equal stressed same-return weights minus base weights
    figure = plot_allocation_shift(
        result=stress_result,
        stress_scenario="85%",
    )

    axis = figure.axes[0]
    bar_widths = np.array([patch.get_width() for patch in axis.patches])

    assert isinstance(figure, Figure)
    assert axis.get_title() == ("Same-Return Allocation Shift: 85% Correlation Floor")

    np.testing.assert_allclose(
        np.sort(bar_widths),
        np.array(
            [
                -0.15,
                0.15,
            ]
        ),
    )

    plt.close(figure)


def test_allocation_shift_raises_when_threshold_removes_all_changes(
    stress_result: CorrelationStressResult,
) -> None:
    # A threshold above every allocation change should prevent an empty chart
    with pytest.raises(
        ValueError,
        match="No allocation changes meet the minimum threshold",
    ):
        plot_allocation_shift(
            result=stress_result,
            stress_scenario="60%",
            minimum_change=0.075,
        )


def test_stress_plot_is_saved(
    stress_result: CorrelationStressResult,
    tmp_path: Path,
) -> None:
    # Saving should create parent directories and write a non-empty image file
    output_path = tmp_path / "figures" / "stress" / "variance.png"

    figure = plot_variance_stress_comparison(
        {
            "Portfolio": stress_result,
        },
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_unknown_stress_scenario_raises_error(
    stress_result: CorrelationStressResult,
) -> None:
    # Allocation shifts require a scenario label present in the weight table
    with pytest.raises(
        ValueError,
        match="Unknown stress scenario",
    ):
        plot_allocation_shift(
            result=stress_result,
            stress_scenario="95%",
        )


def test_base_cannot_be_used_as_stress_scenario(
    stress_result: CorrelationStressResult,
) -> None:
    # Comparing Base with itself would produce no stress-induced allocation change
    with pytest.raises(
        ValueError,
        match="non-base scenario",
    ):
        plot_allocation_shift(
            result=stress_result,
            stress_scenario="Base",
        )


@pytest.mark.parametrize(
    "minimum_change",
    [
        -0.01,
        np.nan,
        np.inf,
    ],
)
def test_invalid_minimum_change_raises_error(
    stress_result: CorrelationStressResult,
    minimum_change: float,
) -> None:
    # Allocation display thresholds must be finite and non-negative
    with pytest.raises(
        ValueError,
        match="minimum_change",
    ):
        plot_allocation_shift(
            result=stress_result,
            stress_scenario="85%",
            minimum_change=minimum_change,
        )


def test_empty_stress_results_raise_error() -> None:
    # Comparison charts require at least one portfolio stress result
    with pytest.raises(
        ValueError,
        match="At least one stress result",
    ):
        plot_variance_stress_comparison({})


def test_missing_summary_column_raises_error(
    stress_result: CorrelationStressResult,
) -> None:
    # Plot validation should fail clearly when required stress metrics are absent
    invalid_summary = stress_result.summary.drop(columns="variance_multiplier")

    invalid_result = CorrelationStressResult(
        base_correlation=stress_result.base_correlation,
        scenarios=stress_result.scenarios,
        summary=invalid_summary,
        efficient_weights=stress_result.efficient_weights,
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        plot_variance_stress_comparison(
            {
                "Portfolio": invalid_result,
            }
        )


def test_duplicate_scenario_labels_raise_error(
    stress_result: CorrelationStressResult,
) -> None:
    # Scenario labels must uniquely identify points across plots and weight tables
    duplicate_summary = stress_result.summary.copy()
    duplicate_summary.loc[
        2,
        "scenario",
    ] = "60%"

    invalid_result = CorrelationStressResult(
        base_correlation=stress_result.base_correlation,
        scenarios=stress_result.scenarios,
        summary=duplicate_summary,
        efficient_weights=stress_result.efficient_weights,
    )

    with pytest.raises(
        ValueError,
        match="duplicate scenario labels",
    ):
        plot_volatility_stress_comparison(
            {
                "Portfolio": invalid_result,
            }
        )
