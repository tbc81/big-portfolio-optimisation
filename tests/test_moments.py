"""Test annual moment estimation from periodic log-return scenarios."""

import numpy as np
import pandas as pd
import pytest

from big_portfolio.estimation.moments import estimate_annual_moments


@pytest.fixture
def log_returns() -> pd.DataFrame:
    """Return representative log-return scenarios for two assets."""
    return pd.DataFrame(
        {
            "Asset A": [
                0.010,
                0.020,
                -0.010,
                0.005,
                0.015,
            ],
            "Asset B": [
                0.000,
                -0.010,
                0.020,
                0.010,
                0.005,
            ],
        }
    )


def test_log_moments_are_annualised_correctly(
    log_returns: pd.DataFrame,
) -> None:
    # Periodic log moments should scale linearly by the periods per year
    periods_per_year = 52

    result = estimate_annual_moments(
        log_returns=log_returns,
        periods_per_year=periods_per_year,
    )

    np.testing.assert_allclose(
        result.annual_log_mean.to_numpy(),
        periods_per_year * result.period_log_mean.to_numpy(),
    )
    np.testing.assert_allclose(
        result.annual_log_covariance.to_numpy(),
        periods_per_year * result.period_log_covariance.to_numpy(),
    )


def test_moment_estimation_matches_lognormal_formula(
    log_returns: pd.DataFrame,
) -> None:
    # Arithmetic moments should match the multivariate lognormal transformation
    periods_per_year = 2

    result = estimate_annual_moments(
        log_returns=log_returns,
        periods_per_year=periods_per_year,
    )

    period_mean = log_returns.mean().to_numpy()
    period_covariance = log_returns.cov().to_numpy()

    annual_mean = periods_per_year * period_mean
    annual_covariance = periods_per_year * period_covariance

    expected_gross_returns = np.exp(annual_mean + 0.5 * np.diag(annual_covariance))
    expected_returns = expected_gross_returns - 1.0

    expected_covariance = np.outer(
        expected_gross_returns,
        expected_gross_returns,
    ) * (np.exp(annual_covariance) - 1.0)

    np.testing.assert_allclose(
        result.expected_returns.to_numpy(),
        expected_returns,
    )
    np.testing.assert_allclose(
        result.covariance.to_numpy(),
        expected_covariance,
    )


def test_asset_labels_are_preserved(
    log_returns: pd.DataFrame,
) -> None:
    # Asset ordering must remain stable across vectors and covariance matrices
    result = estimate_annual_moments(
        log_returns=log_returns,
        periods_per_year=52,
    )

    expected_labels = [
        "Asset A",
        "Asset B",
    ]

    assert result.expected_returns.index.tolist() == expected_labels
    assert result.volatilities.index.tolist() == expected_labels
    assert result.covariance.index.tolist() == expected_labels
    assert result.covariance.columns.tolist() == expected_labels


def test_covariance_is_symmetric_and_positive_definite(
    log_returns: pd.DataFrame,
) -> None:
    # Optimisation requires a symmetric positive-definite covariance matrix
    result = estimate_annual_moments(
        log_returns=log_returns,
        periods_per_year=52,
    )

    covariance = result.covariance.to_numpy()

    np.testing.assert_allclose(
        covariance,
        covariance.T,
    )
    assert result.minimum_eigenvalue > 0.0


def test_reported_minimum_eigenvalue_matches_covariance(
    log_returns: pd.DataFrame,
) -> None:
    # The stored diagnostic should equal the smallest covariance eigenvalue
    result = estimate_annual_moments(
        log_returns=log_returns,
        periods_per_year=52,
    )

    expected_minimum = np.linalg.eigvalsh(result.covariance.to_numpy()).min()

    assert result.minimum_eigenvalue == pytest.approx(expected_minimum)


def test_volatility_matches_covariance_diagonal(
    log_returns: pd.DataFrame,
) -> None:
    # Individual volatility should equal the square root of each asset variance
    result = estimate_annual_moments(
        log_returns=log_returns,
        periods_per_year=52,
    )

    expected_volatility = np.sqrt(np.diag(result.covariance.to_numpy()))

    np.testing.assert_allclose(
        result.volatilities.to_numpy(),
        expected_volatility,
    )


def test_missing_returns_raise_error() -> None:
    # Moment estimation requires complete finite scenarios from the returns layer
    log_returns = pd.DataFrame(
        {
            "Asset A": [
                0.01,
                np.nan,
                0.02,
            ],
            "Asset B": [
                0.01,
                0.02,
                0.03,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        estimate_annual_moments(
            log_returns=log_returns,
            periods_per_year=52,
        )


def test_infinite_return_raises_error() -> None:
    # Infinite observations would contaminate every estimated moment
    log_returns = pd.DataFrame(
        {
            "Asset A": [
                0.01,
                np.inf,
                0.02,
            ],
            "Asset B": [
                0.01,
                0.02,
                0.03,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        estimate_annual_moments(
            log_returns=log_returns,
            periods_per_year=52,
        )


def test_non_numeric_returns_raise_error() -> None:
    # Return scenarios should enter estimation as numerical data
    log_returns = pd.DataFrame(
        {
            "Asset A": [
                "0.01",
                "0.02",
            ],
            "Asset B": [
                "0.03",
                "0.04",
            ],
        }
    )

    with pytest.raises(
        TypeError,
        match="Log-return data must be numeric",
    ):
        estimate_annual_moments(
            log_returns=log_returns,
            periods_per_year=52,
        )


def test_duplicate_asset_names_raise_error() -> None:
    # Duplicate labels would make optimisation vectors and matrix columns ambiguous
    log_returns = pd.DataFrame(
        [
            [0.01, 0.02],
            [0.02, 0.03],
            [0.00, 0.01],
        ],
        columns=[
            "Asset A",
            "Asset A",
        ],
    )

    with pytest.raises(
        ValueError,
        match="Asset names must be unique",
    ):
        estimate_annual_moments(
            log_returns=log_returns,
            periods_per_year=52,
        )


def test_single_scenario_raises_error() -> None:
    # Sample covariance estimation needs at least two return scenarios
    log_returns = pd.DataFrame(
        {
            "Asset A": [0.01],
            "Asset B": [0.02],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        estimate_annual_moments(
            log_returns=log_returns,
            periods_per_year=52,
        )


def test_constant_returns_raise_positive_definiteness_error() -> None:
    # Zero return variation produces a singular covariance unsuitable for Cholesky
    log_returns = pd.DataFrame(
        {
            "Asset A": [
                0.01,
                0.01,
                0.01,
            ],
            "Asset B": [
                0.02,
                0.02,
                0.02,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="positive definite",
    ):
        estimate_annual_moments(
            log_returns=log_returns,
            periods_per_year=52,
        )


def test_invalid_periods_per_year_raises_error(
    log_returns: pd.DataFrame,
) -> None:
    # Annualisation requires a strictly positive number of periods per year
    with pytest.raises(
        ValueError,
        match="periods_per_year",
    ):
        estimate_annual_moments(
            log_returns=log_returns,
            periods_per_year=0,
        )
