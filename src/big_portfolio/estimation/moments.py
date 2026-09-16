"""Estimate annual arithmetic moments from periodic log-return scenarios."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass(frozen=True)
class MomentEstimate:
    """Store periodic, annualised and arithmetic return-moment estimates."""

    period_log_mean: pd.Series
    period_log_covariance: pd.DataFrame
    annual_log_mean: pd.Series
    annual_log_covariance: pd.DataFrame
    expected_returns: pd.Series
    covariance: pd.DataFrame
    volatilities: pd.Series
    minimum_eigenvalue: float


def estimate_annual_moments(
    log_returns: pd.DataFrame,
    periods_per_year: int,
) -> MomentEstimate:
    """Estimate annual arithmetic moments from periodic log returns."""
    _validate_log_returns(
        log_returns=log_returns,
        periods_per_year=periods_per_year,
    )

    period_log_mean = log_returns.mean()
    period_log_covariance = log_returns.cov()

    # Independent log returns aggregate through time by summing means and covariances
    annual_log_mean = periods_per_year * period_log_mean
    annual_log_covariance = periods_per_year * period_log_covariance

    annual_log_mean_values = annual_log_mean.to_numpy(dtype=float)
    annual_log_covariance_values = annual_log_covariance.to_numpy(dtype=float)
    log_variances = np.diag(annual_log_covariance_values)

    # E[1 + R_i] = exp(mu_i + 0.5 * Sigma_ii)
    expected_gross_returns = np.exp(annual_log_mean_values + 0.5 * log_variances)

    expected_returns = pd.Series(
        expected_gross_returns - 1.0,
        index=log_returns.columns,
        name="annual_expected_return",
    )

    # Cov(R_i, R_j) = E[1+R_i]E[1+R_j](exp(Sigma_ij) - 1)
    covariance_values = np.outer(
        expected_gross_returns,
        expected_gross_returns,
    ) * (np.exp(annual_log_covariance_values) - 1.0)

    covariance = pd.DataFrame(
        covariance_values,
        index=log_returns.columns,
        columns=log_returns.columns,
    )

    minimum_eigenvalue = _validate_covariance(covariance)

    volatilities = pd.Series(
        np.sqrt(np.diag(covariance_values)),
        index=log_returns.columns,
        name="annual_volatility",
    )

    return MomentEstimate(
        period_log_mean=period_log_mean,
        period_log_covariance=period_log_covariance,
        annual_log_mean=annual_log_mean,
        annual_log_covariance=annual_log_covariance,
        expected_returns=expected_returns,
        covariance=covariance,
        volatilities=volatilities,
        minimum_eigenvalue=minimum_eigenvalue,
    )


def _validate_log_returns(
    log_returns: pd.DataFrame,
    periods_per_year: int,
) -> None:
    """Validate return scenarios before moment estimation."""
    if log_returns.empty:
        raise ValueError("Log-return data cannot be empty.")

    if len(log_returns) < 2:
        raise ValueError("At least two return scenarios are required.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if not log_returns.columns.is_unique:
        raise ValueError("Asset names must be unique.")

    non_numeric_columns = [
        column
        for column in log_returns.columns
        if not is_numeric_dtype(log_returns[column].dtype)
    ]

    if non_numeric_columns:
        raise TypeError(
            f"Log-return data must be numeric for: {sorted(non_numeric_columns)}"
        )

    values = log_returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Log-return data must contain only finite values.")


def _validate_covariance(
    covariance: pd.DataFrame,
) -> float:
    """Validate covariance for optimisation and return its minimum eigenvalue."""
    values = covariance.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Covariance matrix must contain only finite values.")

    if not np.allclose(
        values,
        values.T,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Covariance matrix is not symmetric.")

    minimum_eigenvalue = float(np.linalg.eigvalsh(values).min())

    # Cholesky factorisation used by the optimiser requires positive definiteness
    if minimum_eigenvalue <= 0.0:
        raise ValueError("Covariance matrix is not positive definite.")

    return minimum_eigenvalue
