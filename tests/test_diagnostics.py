"""Test fixed-portfolio statistics and model-implied efficiency diagnostics."""

import numpy as np
import pandas as pd
import pytest

from big_portfolio.analysis.diagnostics import (
    analyse_portfolio_efficiency,
    build_allocation_comparison,
    calculate_portfolio_snapshot,
)

SOLVER_TOLERANCE = 1e-7


@pytest.fixture
def expected_returns() -> pd.Series:
    """Return expected annual returns for two risky assets."""
    return pd.Series(
        {
            "Asset A": 0.10,
            "Asset B": 0.20,
        }
    )


@pytest.fixture
def covariance() -> pd.DataFrame:
    """Return a positive-definite annual covariance matrix."""
    return pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=[
            "Asset A",
            "Asset B",
        ],
        columns=[
            "Asset A",
            "Asset B",
        ],
    )


@pytest.fixture
def current_weights() -> pd.Series:
    """Return risky weights with a ten-percent cash allocation."""
    return pd.Series(
        {
            "Asset A": 0.60,
            "Asset B": 0.30,
        }
    )


def test_portfolio_snapshot_matches_formula(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
) -> None:
    # Snapshot statistics should match the portfolio return and variance formulas
    result = calculate_portfolio_snapshot(
        expected_returns=expected_returns,
        covariance=covariance,
        weights=current_weights,
        cash_weight=0.10,
    )

    weight_values = current_weights.to_numpy()
    expected_return_values = expected_returns.to_numpy()
    covariance_values = covariance.to_numpy()

    expected_portfolio_return = float(weight_values @ expected_return_values)
    expected_volatility = float(
        np.sqrt(weight_values @ covariance_values @ weight_values)
    )

    assert result.expected_return == pytest.approx(expected_portfolio_return)
    assert result.volatility == pytest.approx(expected_volatility)


def test_cash_return_is_included(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
) -> None:
    # Cash should contribute its configured return to the total portfolio return
    cash_weight = 0.10
    cash_expected_return = 0.05

    result = calculate_portfolio_snapshot(
        expected_returns=expected_returns,
        covariance=covariance,
        weights=current_weights,
        cash_weight=cash_weight,
        cash_expected_return=cash_expected_return,
    )

    risky_return = float(current_weights.to_numpy() @ expected_returns.to_numpy())
    expected_total_return = risky_return + cash_weight * cash_expected_return

    assert result.expected_return == pytest.approx(expected_total_return)


def test_input_labels_are_aligned_before_calculation(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
) -> None:
    # Label alignment should make input ordering irrelevant to portfolio statistics
    reversed_assets = [
        "Asset B",
        "Asset A",
    ]

    reordered_covariance = covariance.loc[
        reversed_assets,
        reversed_assets,
    ]
    reordered_weights = current_weights.loc[reversed_assets]

    original = calculate_portfolio_snapshot(
        expected_returns=expected_returns,
        covariance=covariance,
        weights=current_weights,
        cash_weight=0.10,
    )
    reordered = calculate_portfolio_snapshot(
        expected_returns=expected_returns,
        covariance=reordered_covariance,
        weights=reordered_weights,
        cash_weight=0.10,
    )

    assert reordered.expected_return == pytest.approx(original.expected_return)
    assert reordered.volatility == pytest.approx(original.volatility)
    assert reordered.weights.index.tolist() == expected_returns.index.tolist()


def test_weights_must_sum_to_one(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # Risky weights and cash must account for the full portfolio budget
    invalid_weights = pd.Series(
        {
            "Asset A": 0.40,
            "Asset B": 0.30,
        }
    )

    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        calculate_portfolio_snapshot(
            expected_returns=expected_returns,
            covariance=covariance,
            weights=invalid_weights,
            cash_weight=0.10,
        )


def test_negative_portfolio_weight_raises_error(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # Fixed portfolios follow the same long-only assumption as the optimiser
    weights = pd.Series(
        {
            "Asset A": 1.10,
            "Asset B": -0.10,
        }
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        calculate_portfolio_snapshot(
            expected_returns=expected_returns,
            covariance=covariance,
            weights=weights,
            cash_weight=0.0,
        )


def test_asymmetric_covariance_raises_error(
    expected_returns: pd.Series,
    current_weights: pd.Series,
) -> None:
    # Portfolio variance requires a symmetric covariance matrix
    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.02, 0.09],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    with pytest.raises(
        ValueError,
        match="must be symmetric",
    ):
        calculate_portfolio_snapshot(
            expected_returns=expected_returns,
            covariance=covariance,
            weights=current_weights,
            cash_weight=0.10,
        )


def test_singular_covariance_can_value_fixed_portfolio() -> None:
    # A positive-semidefinite covariance is sufficient for fixed-portfolio variance
    expected_returns = pd.Series(
        {
            "Asset A": 0.10,
            "Asset B": 0.10,
        }
    )
    covariance = pd.DataFrame(
        [
            [0.04, 0.04],
            [0.04, 0.04],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    weights = pd.Series(
        {
            "Asset A": 0.50,
            "Asset B": 0.50,
        }
    )

    result = calculate_portfolio_snapshot(
        expected_returns=expected_returns,
        covariance=covariance,
        weights=weights,
        cash_weight=0.0,
    )

    assert result.volatility == pytest.approx(0.20)


def test_non_finite_expected_return_raises_error(
    covariance: pd.DataFrame,
    current_weights: pd.Series,
) -> None:
    # Non-finite return estimates would invalidate portfolio statistics
    expected_returns = pd.Series(
        {
            "Asset A": 0.10,
            "Asset B": np.inf,
        }
    )

    with pytest.raises(
        ValueError,
        match="Expected returns must be finite",
    ):
        calculate_portfolio_snapshot(
            expected_returns=expected_returns,
            covariance=covariance,
            weights=current_weights,
            cash_weight=0.10,
        )


@pytest.mark.mosek
def test_efficiency_comparison_improves_objectives(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
) -> None:
    # Efficient alternatives should improve chosen objective at the same constraint
    result = analyse_portfolio_efficiency(
        expected_returns=expected_returns,
        covariance=covariance,
        current_weights=current_weights,
        current_cash_weight=0.10,
    )

    assert result.same_risk.volatility <= result.current.volatility + SOLVER_TOLERANCE
    assert (
        result.same_risk.expected_return
        >= result.current.expected_return - SOLVER_TOLERANCE
    )

    assert (
        result.same_return.expected_return
        >= result.current.expected_return - SOLVER_TOLERANCE
    )
    assert result.same_return.volatility <= result.current.volatility + SOLVER_TOLERANCE


@pytest.mark.mosek
def test_efficiency_improvements_match_reported_diagnostics(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
) -> None:
    # Reported improvements should equal the differences between portfolio solutions
    result = analyse_portfolio_efficiency(
        expected_returns=expected_returns,
        covariance=covariance,
        current_weights=current_weights,
        current_cash_weight=0.10,
    )

    expected_return_improvement = (
        result.same_risk.expected_return - result.current.expected_return
    )
    expected_volatility_reduction = (
        result.current.volatility - result.same_return.volatility
    )

    assert result.return_improvement == pytest.approx(expected_return_improvement)
    assert result.volatility_reduction == pytest.approx(expected_volatility_reduction)


@pytest.mark.mosek
def test_allocation_comparison_includes_cash(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
) -> None:
    # Allocation output should preserve risky asset order and include cash last
    result = analyse_portfolio_efficiency(
        expected_returns=expected_returns,
        covariance=covariance,
        current_weights=current_weights,
        current_cash_weight=0.10,
    )

    comparison = build_allocation_comparison(result)

    assert comparison.index.tolist() == [
        "Asset A",
        "Asset B",
        "Cash",
    ]
    assert comparison.columns.tolist() == [
        "current",
        "same_risk",
        "same_return",
    ]

    np.testing.assert_allclose(
        comparison.sum(axis=0).to_numpy(),
        np.ones(3),
        atol=SOLVER_TOLERANCE,
    )
