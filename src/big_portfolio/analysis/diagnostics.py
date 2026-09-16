"""Calculate portfolio statistics and model-implied efficiency diagnostics."""

from dataclasses import dataclass
from math import fsum, isclose, isfinite

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from big_portfolio.optimisation.mean_variance import (
    PortfolioSolution,
    maximise_return_at_risk,
    minimise_risk_at_return,
)

COVARIANCE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Store a fixed portfolio allocation and its modelled statistics."""

    weights: pd.Series
    cash_weight: float
    expected_return: float
    volatility: float


@dataclass(frozen=True)
class PortfolioDiagnosticResult:
    """Store portfolio statistics and efficiency comparisons."""

    current: PortfolioSnapshot
    same_risk: PortfolioSolution
    same_return: PortfolioSolution
    return_improvement: float
    volatility_reduction: float


def analyse_portfolio_efficiency(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
    current_cash_weight: float,
    cash_expected_return: float = 0.0,
) -> PortfolioDiagnosticResult:
    """Compare a portfolio with same-risk and same-return efficient alternatives."""
    current = calculate_portfolio_snapshot(
        expected_returns=expected_returns,
        covariance=covariance,
        weights=current_weights,
        cash_weight=current_cash_weight,
        cash_expected_return=cash_expected_return,
    )

    # Hold portfolio volatility fixed and maximise expected return
    same_risk = maximise_return_at_risk(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_limit=current.volatility,
        cash_expected_return=cash_expected_return,
    )

    # Hold portfolio expected return fixed and minimise volatility
    same_return = minimise_risk_at_return(
        expected_returns=expected_returns,
        covariance=covariance,
        target_return=current.expected_return,
        cash_expected_return=cash_expected_return,
    )

    return PortfolioDiagnosticResult(
        current=current,
        same_risk=same_risk,
        same_return=same_return,
        return_improvement=(same_risk.expected_return - current.expected_return),
        volatility_reduction=(current.volatility - same_return.volatility),
    )


def calculate_portfolio_snapshot(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: pd.Series,
    cash_weight: float,
    cash_expected_return: float = 0.0,
) -> PortfolioSnapshot:
    """Calculate expected return and volatility for a fixed portfolio."""
    (
        aligned_returns,
        aligned_covariance,
        aligned_weights,
    ) = _prepare_portfolio_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        weights=weights,
    )

    _validate_cash_weight(cash_weight)

    if not isfinite(cash_expected_return):
        raise ValueError("cash_expected_return must be finite.")

    weight_values = aligned_weights.to_numpy(dtype=float)
    return_values = aligned_returns.to_numpy(dtype=float)
    covariance_values = aligned_covariance.to_numpy(dtype=float)

    # Portfolio capital must be fully allocated across risky assets and cash
    total_weight = fsum(
        [
            *weight_values,
            cash_weight,
        ]
    )

    if not isclose(
        total_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("Portfolio weights including cash must sum to 1.0.")

    # E[R_p] = x^T mu + c * r_cash
    portfolio_return = float(
        weight_values @ return_values + cash_weight * cash_expected_return
    )

    # Portfolio variance = x^T Sigma x
    portfolio_variance = float(weight_values @ covariance_values @ weight_values)

    # Floating-point arithmetic can produce tiny negative values around zero
    portfolio_volatility = float(
        np.sqrt(
            max(
                portfolio_variance,
                0.0,
            )
        )
    )

    return PortfolioSnapshot(
        weights=aligned_weights,
        cash_weight=float(cash_weight),
        expected_return=portfolio_return,
        volatility=portfolio_volatility,
    )


def build_allocation_comparison(
    result: PortfolioDiagnosticResult,
) -> pd.DataFrame:
    """Build a table comparing observed and efficient allocations."""
    asset_names = result.current.weights.index.tolist()

    comparison = pd.DataFrame(
        {
            "current": result.current.weights,
            "same_risk": result.same_risk.weights,
            "same_return": result.same_return.weights,
        },
        index=asset_names,
    )

    # Add cash after the risky assets for a complete allocation comparison
    comparison.loc["Cash"] = [
        result.current.cash_weight,
        result.same_risk.cash_weight,
        result.same_return.cash_weight,
    ]

    return comparison


def _prepare_portfolio_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: pd.Series,
) -> tuple[
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Validate and align portfolio statistics and weights."""
    if expected_returns.empty:
        raise ValueError("Expected returns cannot be empty.")

    if covariance.empty:
        raise ValueError("Covariance matrix cannot be empty.")

    if weights.empty:
        raise ValueError("Portfolio weights cannot be empty.")

    if not expected_returns.index.is_unique:
        raise ValueError("Expected-return asset names must be unique.")

    if not weights.index.is_unique:
        raise ValueError("Portfolio weight asset names must be unique.")

    if not covariance.index.is_unique or not covariance.columns.is_unique:
        raise ValueError("Covariance asset names must be unique.")

    if not is_numeric_dtype(expected_returns.dtype):
        raise TypeError("Expected returns must be numeric.")

    if not is_numeric_dtype(weights.dtype):
        raise TypeError("Portfolio weights must be numeric.")

    non_numeric_covariance = [
        column
        for column in covariance.columns
        if not is_numeric_dtype(covariance[column].dtype)
    ]

    if non_numeric_covariance:
        raise TypeError(
            f"Covariance values must be numeric for: {sorted(non_numeric_covariance)}"
        )

    asset_names = expected_returns.index.tolist()
    asset_set = set(asset_names)

    if set(weights.index) != asset_set:
        raise ValueError(
            "Portfolio weights and expected returns must contain the same assets."
        )

    if set(covariance.index) != asset_set or set(covariance.columns) != asset_set:
        raise ValueError(
            "Covariance and expected returns must contain the same assets."
        )

    aligned_returns = expected_returns.loc[asset_names]
    aligned_weights = weights.loc[asset_names]
    aligned_covariance = covariance.loc[
        asset_names,
        asset_names,
    ]

    return_values = aligned_returns.to_numpy(dtype=float)
    weight_values = aligned_weights.to_numpy(dtype=float)
    covariance_values = aligned_covariance.to_numpy(dtype=float)

    if not np.isfinite(return_values).all():
        raise ValueError("Expected returns must be finite.")

    if not np.isfinite(weight_values).all():
        raise ValueError("Portfolio weights must be finite.")

    if not np.isfinite(covariance_values).all():
        raise ValueError("Covariance values must be finite.")

    if (weight_values < 0.0).any():
        raise ValueError("Portfolio weights cannot be negative.")

    if not np.allclose(
        covariance_values,
        covariance_values.T,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Covariance matrix must be symmetric.")

    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance_values).min())

    # Fixed-portfolio variance requires a positive-semidefinite covariance matrix
    if minimum_eigenvalue < -COVARIANCE_TOLERANCE:
        raise ValueError("Covariance matrix must be positive semidefinite.")

    return (
        aligned_returns,
        aligned_covariance,
        aligned_weights,
    )


def _validate_cash_weight(
    cash_weight: float,
) -> None:
    """Validate the portfolio cash allocation."""
    if not isfinite(cash_weight):
        raise ValueError("cash_weight must be finite.")

    if cash_weight < 0.0:
        raise ValueError("cash_weight cannot be negative.")
