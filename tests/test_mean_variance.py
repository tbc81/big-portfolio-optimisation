"""Test MOSEK mean-variance portfolio optimisation."""

import numpy as np
import pandas as pd
import pytest

from big_portfolio.optimisation.mean_variance import (
    generate_efficient_frontier,
    maximise_return_at_risk,
    minimise_risk_at_return,
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


@pytest.mark.mosek
def test_maximise_return_respects_risk_limit(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # The fixed-risk model must remain fully invested, long-only and within its limit
    risk_limit = 0.15

    result = maximise_return_at_risk(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_limit=risk_limit,
    )

    total_weight = result.weights.sum() + result.cash_weight

    assert total_weight == pytest.approx(
        1.0,
        abs=SOLVER_TOLERANCE,
    )
    assert (result.weights.to_numpy() >= -SOLVER_TOLERANCE).all()
    assert result.cash_weight >= -SOLVER_TOLERANCE
    assert result.volatility <= risk_limit + SOLVER_TOLERANCE


@pytest.mark.mosek
def test_minimise_risk_respects_return_floor(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # The fixed-return model must achieve its target using a fully invested portfolio
    target_return = 0.12

    result = minimise_risk_at_return(
        expected_returns=expected_returns,
        covariance=covariance,
        target_return=target_return,
    )

    total_weight = result.weights.sum() + result.cash_weight

    assert result.expected_return >= target_return - SOLVER_TOLERANCE
    assert total_weight == pytest.approx(
        1.0,
        abs=SOLVER_TOLERANCE,
    )
    assert (result.weights.to_numpy() >= -SOLVER_TOLERANCE).all()
    assert result.cash_weight >= -SOLVER_TOLERANCE


@pytest.mark.mosek
def test_reported_statistics_match_returned_allocation(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # Reported return and volatility should agree with the returned portfolio weights
    result = maximise_return_at_risk(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_limit=0.20,
    )

    weights = result.weights.to_numpy()

    expected_return = float(weights @ expected_returns.to_numpy())
    expected_volatility = float(np.sqrt(weights @ covariance.to_numpy() @ weights))

    assert result.expected_return == pytest.approx(expected_return)
    assert result.volatility == pytest.approx(expected_volatility)


@pytest.mark.mosek
def test_efficient_frontier_is_fully_invested_and_long_only(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # Every frontier portfolio should satisfy the same budget and long-only constraints
    risk_aversions = [
        10.0,
        1.0,
        0.1,
    ]

    result = generate_efficient_frontier(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_aversions=risk_aversions,
    )

    total_weights = result.weights.sum(axis=1)
    risky_weights = result.weights.drop(columns="Cash")

    assert len(result.summary) == len(risk_aversions)
    assert len(result.weights) == len(risk_aversions)

    np.testing.assert_allclose(
        total_weights.to_numpy(),
        np.ones(len(risk_aversions)),
        atol=SOLVER_TOLERANCE,
    )
    assert (risky_weights.to_numpy() >= -SOLVER_TOLERANCE).all()
    assert (result.weights["Cash"].to_numpy() >= -SOLVER_TOLERANCE).all()


@pytest.mark.mosek
def test_frontier_preserves_risk_aversion_order(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # Frontier results should remain aligned with the caller's parameter sequence
    risk_aversions = [
        5.0,
        0.5,
        2.0,
    ]

    result = generate_efficient_frontier(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_aversions=risk_aversions,
    )

    np.testing.assert_allclose(
        result.summary["risk_aversion"].to_numpy(),
        np.array(risk_aversions),
    )


@pytest.mark.mosek
def test_lower_risk_aversion_accepts_more_risk(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # A weaker volatility penalty should not reduce optimal portfolio volatility
    result = generate_efficient_frontier(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_aversions=[
            10.0,
            1.0,
            0.1,
        ],
    )

    volatility = result.summary["volatility"].to_numpy()

    assert (np.diff(volatility) >= -1e-7).all()


@pytest.mark.mosek
def test_covariance_is_aligned_to_expected_return_order(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # Label alignment should make covariance row order irrelevant to the solution
    reversed_assets = [
        "Asset B",
        "Asset A",
    ]
    reordered_covariance = covariance.loc[
        reversed_assets,
        reversed_assets,
    ]

    original_result = minimise_risk_at_return(
        expected_returns=expected_returns,
        covariance=covariance,
        target_return=0.12,
    )
    reordered_result = minimise_risk_at_return(
        expected_returns=expected_returns,
        covariance=reordered_covariance,
        target_return=0.12,
    )

    np.testing.assert_allclose(
        reordered_result.weights.to_numpy(),
        original_result.weights.to_numpy(),
        atol=1e-6,
    )
    assert reordered_result.expected_return == pytest.approx(
        original_result.expected_return,
        abs=SOLVER_TOLERANCE,
    )
    assert reordered_result.volatility == pytest.approx(
        original_result.volatility,
        abs=SOLVER_TOLERANCE,
    )


@pytest.mark.mosek
def test_cash_is_selected_when_it_has_highest_expected_return() -> None:
    # Cash should receive the allocation when it dominates every risky asset
    expected_returns = pd.Series(
        {
            "Asset A": -0.01,
            "Asset B": -0.02,
        }
    )
    covariance = pd.DataFrame(
        [
            [0.04, 0.00],
            [0.00, 0.09],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    result = maximise_return_at_risk(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_limit=0.50,
        cash_expected_return=0.03,
    )

    assert result.cash_weight == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert np.abs(result.weights.to_numpy()).max() <= 1e-6
    assert result.expected_return == pytest.approx(
        0.03,
        abs=1e-6,
    )


def test_non_positive_definite_covariance_raises_error(
    expected_returns: pd.Series,
) -> None:
    # Cholesky-based risk modelling requires a positive-definite covariance matrix
    covariance = pd.DataFrame(
        [
            [0.04, 0.05],
            [0.05, 0.04],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    with pytest.raises(
        ValueError,
        match="positive definite",
    ):
        maximise_return_at_risk(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_limit=0.20,
        )


def test_mismatched_asset_labels_raise_error(
    expected_returns: pd.Series,
) -> None:
    # Expected-return and covariance labels must describe the same asset universe
    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=[
            "Asset A",
            "Asset C",
        ],
        columns=[
            "Asset A",
            "Asset C",
        ],
    )

    with pytest.raises(
        ValueError,
        match="must contain the same assets",
    ):
        maximise_return_at_risk(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_limit=0.20,
        )


@pytest.mark.parametrize(
    ("risk_aversions", "error_message"),
    [
        ([-0.1], "cannot be negative"),
        ([np.nan], "must be finite"),
        ([np.inf], "must be finite"),
    ],
)
def test_invalid_risk_aversion_raises_error(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversions: list[float],
    error_message: str,
) -> None:
    # Frontier parameters must be finite and non-negative before a model is solved
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        generate_efficient_frontier(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_aversions=risk_aversions,
        )


def test_negative_risk_limit_raises_error(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    # Volatility limits cannot be negative under the portfolio-risk definition
    with pytest.raises(
        ValueError,
        match="risk_limit",
    ):
        maximise_return_at_risk(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_limit=-0.01,
        )
