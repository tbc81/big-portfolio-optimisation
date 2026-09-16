"""Test correlation-matrix stress construction and portfolio sensitivity."""

import numpy as np
import pandas as pd
import pytest

from big_portfolio.analysis.correlation_stress import (
    average_within_group_correlation,
    calculate_group_exposure,
    correlation_to_covariance,
    covariance_to_correlation,
    run_correlation_stress,
    solve_stressed_correlation,
)

SOLVER_TOLERANCE = 1e-7


@pytest.fixture
def covariance() -> pd.DataFrame:
    """Return a positive-definite covariance matrix for three assets."""
    return pd.DataFrame(
        [
            [0.0400, 0.0120, 0.0040],
            [0.0120, 0.0900, 0.0060],
            [0.0040, 0.0060, 0.0100],
        ],
        index=[
            "Asset A",
            "Asset B",
            "Asset C",
        ],
        columns=[
            "Asset A",
            "Asset B",
            "Asset C",
        ],
    )


@pytest.fixture
def expected_returns() -> pd.Series:
    """Return annual expected returns for the stress-test assets."""
    return pd.Series(
        {
            "Asset A": 0.08,
            "Asset B": 0.12,
            "Asset C": 0.05,
        }
    )


@pytest.fixture
def portfolio_weights() -> pd.Series:
    """Return a fully invested risky-asset allocation."""
    return pd.Series(
        {
            "Asset A": 0.40,
            "Asset B": 0.40,
            "Asset C": 0.20,
        }
    )


def test_covariance_to_correlation_has_expected_values(
    covariance: pd.DataFrame,
) -> None:
    # Conversion should preserve labels and produce unit diagonal correlations
    correlation = covariance_to_correlation(covariance)

    np.testing.assert_allclose(
        np.diag(correlation.to_numpy()),
        np.ones(3),
    )
    assert correlation.loc[
        "Asset A",
        "Asset B",
    ] == pytest.approx(0.20)
    assert correlation.index.tolist() == covariance.index.tolist()
    assert correlation.columns.tolist() == covariance.columns.tolist()


def test_covariance_correlation_round_trip(
    covariance: pd.DataFrame,
) -> None:
    # Rebuilding covariance should recover original matrix when correlation unchanged
    correlation = covariance_to_correlation(covariance)

    rebuilt_covariance = correlation_to_covariance(
        correlation=correlation,
        base_covariance=covariance,
    )

    np.testing.assert_allclose(
        rebuilt_covariance.to_numpy(),
        covariance.to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.mosek
def test_stressed_correlation_meets_target(
    covariance: pd.DataFrame,
) -> None:
    # Every selected asset pair should satisfy the configured correlation floor
    base_correlation = covariance_to_correlation(covariance)

    groups = {
        "test_group": [
            "Asset A",
            "Asset B",
        ]
    }

    stressed = solve_stressed_correlation(
        base_correlation=base_correlation,
        groups=groups,
        target_correlation=0.80,
    )

    assert (
        stressed.loc[
            "Asset A",
            "Asset B",
        ]
        >= 0.80 - SOLVER_TOLERANCE
    )


@pytest.mark.mosek
def test_stressed_correlation_remains_positive_definite(
    covariance: pd.DataFrame,
) -> None:
    # The stressed matrix must remain suitable for downstream Cholesky factorisation
    base_correlation = covariance_to_correlation(covariance)

    groups = {
        "test_group": [
            "Asset A",
            "Asset B",
        ]
    }

    stressed = solve_stressed_correlation(
        base_correlation=base_correlation,
        groups=groups,
        target_correlation=0.85,
    )

    eigenvalues = np.linalg.eigvalsh(stressed.to_numpy())

    assert eigenvalues.min() > 0.0


@pytest.mark.mosek
def test_stress_preserves_asset_variances(
    covariance: pd.DataFrame,
) -> None:
    # Correlation-only stress should leave each asset's marginal variance unchanged
    base_correlation = covariance_to_correlation(covariance)

    groups = {
        "test_group": [
            "Asset A",
            "Asset B",
        ]
    }

    stressed_correlation = solve_stressed_correlation(
        base_correlation=base_correlation,
        groups=groups,
        target_correlation=0.85,
    )

    stressed_covariance = correlation_to_covariance(
        correlation=stressed_correlation,
        base_covariance=covariance,
    )

    np.testing.assert_allclose(
        np.diag(stressed_covariance.to_numpy()),
        np.diag(covariance.to_numpy()),
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.mosek
def test_inactive_negative_floor_leaves_correlation_unchanged(
    covariance: pd.DataFrame,
) -> None:
    # A floor below the empirical pair correlation should require no matrix adjustment
    base_correlation = covariance_to_correlation(covariance)

    groups = {
        "test_group": [
            "Asset A",
            "Asset B",
        ]
    }

    stressed = solve_stressed_correlation(
        base_correlation=base_correlation,
        groups=groups,
        target_correlation=-0.10,
    )

    np.testing.assert_allclose(
        stressed.to_numpy(),
        base_correlation.to_numpy(),
        atol=SOLVER_TOLERANCE,
    )


def test_overlapping_groups_count_each_pair_once() -> None:
    # Overlapping classifications should not give repeated asset pairs extra weight
    correlation = pd.DataFrame(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ],
        index=[
            "Asset A",
            "Asset B",
            "Asset C",
        ],
        columns=[
            "Asset A",
            "Asset B",
            "Asset C",
        ],
    )

    groups = {
        "group_one": [
            "Asset A",
            "Asset B",
            "Asset C",
        ],
        "group_two": [
            "Asset A",
            "Asset B",
        ],
    }

    realised_average = average_within_group_correlation(
        correlation=correlation,
        groups=groups,
    )

    expected_average = np.mean(
        [
            0.9,
            0.2,
            0.4,
        ]
    )

    assert realised_average == pytest.approx(expected_average)


@pytest.mark.mosek
def test_portfolio_stress_reports_consistent_risk_effects(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    portfolio_weights: pd.Series,
) -> None:
    # Portfolio-level output should compare stressed risk with the same base allocation
    result = run_correlation_stress(
        expected_returns=expected_returns,
        covariance=covariance,
        current_weights=portfolio_weights,
        current_cash_weight=0.0,
        groups={
            "test_group": [
                "Asset A",
                "Asset B",
            ]
        },
        target_correlations=[0.80],
    )

    base_scenario = result.scenarios[0]
    stressed_scenario = result.scenarios[1]

    assert base_scenario.label == "Base"
    assert stressed_scenario.label == "80%"

    assert base_scenario.current_variance_multiplier == pytest.approx(1.0)
    assert stressed_scenario.current_volatility > base_scenario.current_volatility
    assert stressed_scenario.current_variance_multiplier > 1.0

    assert result.summary.loc[
        0,
        "volatility_change",
    ] == pytest.approx(0.0)


@pytest.mark.mosek
def test_stressed_efficient_portfolio_preserves_return_target(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    portfolio_weights: pd.Series,
) -> None:
    # Each efficient stress portfolio should retain the original portfolio return floor
    result = run_correlation_stress(
        expected_returns=expected_returns,
        covariance=covariance,
        current_weights=portfolio_weights,
        current_cash_weight=0.0,
        groups={
            "test_group": [
                "Asset A",
                "Asset B",
            ]
        },
        target_correlations=[0.80],
    )

    base_return = float(
        portfolio_weights @ expected_returns.loc[portfolio_weights.index]
    )

    for scenario in result.scenarios:
        assert (
            scenario.efficient_solution.expected_return
            >= base_return - SOLVER_TOLERANCE
        )


def test_target_correlations_must_be_strictly_increasing(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    portfolio_weights: pd.Series,
) -> None:
    # Target correlation floors must be strictly increasing
    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        run_correlation_stress(
            expected_returns=expected_returns,
            covariance=covariance,
            current_weights=portfolio_weights,
            current_cash_weight=0.0,
            groups={
                "test_group": [
                    "Asset A",
                    "Asset B",
                ]
            },
            target_correlations=[
                0.80,
                0.60,
            ],
        )


def test_group_exposure_sums_active_weights() -> None:
    # Group exposure should aggregate only holdings represented in the supplied weights
    weights = pd.Series(
        {
            "Asset A": 0.40,
            "Asset B": 0.35,
            "Asset C": 0.25,
        }
    )

    groups = {
        "group_one": [
            "Asset A",
            "Asset B",
        ],
        "group_two": [
            "Asset A",
            "Asset C",
            "Asset D",
        ],
    }

    exposures = calculate_group_exposure(
        weights=weights,
        groups=groups,
    )

    assert exposures["group_one"] == pytest.approx(0.75)
    assert exposures["group_two"] == pytest.approx(0.65)
