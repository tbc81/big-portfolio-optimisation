"""Solve long-only mean-variance portfolio problems with MOSEK Fusion."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd
from mosek.fusion import (
    Domain,
    Expr,
    Expression,
    Model,
    ObjectiveSense,
    SolutionStatus,
    Variable,
)
from pandas.api.types import is_numeric_dtype


@dataclass(frozen=True)
class PortfolioSolution:
    """Store one optimal portfolio and its resulting statistics."""

    weights: pd.Series
    cash_weight: float
    expected_return: float
    volatility: float


@dataclass(frozen=True)
class EfficientFrontierResult:
    """Store efficient-frontier statistics and portfolio allocations."""

    summary: pd.DataFrame
    weights: pd.DataFrame


def maximise_return_at_risk(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_limit: float,
    cash_expected_return: float = 0.0,
) -> PortfolioSolution:
    """Maximise expected return subject to an annual volatility limit."""
    (
        asset_names,
        expected_values,
        covariance_values,
        cholesky_factor,
    ) = _prepare_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
    )

    _validate_scalar(
        risk_limit,
        name="risk_limit",
        minimum=0.0,
    )
    _validate_scalar(
        cash_expected_return,
        name="cash_expected_return",
    )

    with Model("MaximiseReturnAtRisk") as model:
        weights = model.variable(
            "weights",
            len(asset_names),
            Domain.greaterThan(0.0),
        )
        cash_weight = model.variable(
            "cash_weight",
            Domain.greaterThan(0.0),
        )

        # Risky assets and cash must exhaust the available portfolio capital
        model.constraint(
            "budget",
            Expr.add(
                Expr.sum(weights),
                cash_weight,
            ),
            Domain.equalsTo(1.0),
        )

        # If Sigma = GG^T, portfolio volatility is ||G^T x||_2
        model.constraint(
            "risk",
            Expr.vstack(
                risk_limit,
                Expr.mul(
                    cholesky_factor.T,
                    weights,
                ),
            ),
            Domain.inQCone(),
        )

        portfolio_return = _portfolio_return_expression(
            weights=weights,
            cash_weight=cash_weight,
            expected_returns=expected_values,
            cash_expected_return=cash_expected_return,
        )

        model.objective(
            "expected_return",
            ObjectiveSense.Maximize,
            portfolio_return,
        )

        model.solve()
        _require_optimal_solution(model)

        return _build_solution(
            asset_names=asset_names,
            weights=np.asarray(
                weights.level(),
                dtype=float,
            ),
            cash_weight=float(cash_weight.level()[0]),
            expected_returns=expected_values,
            covariance=covariance_values,
            cash_expected_return=cash_expected_return,
        )


def minimise_risk_at_return(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    cash_expected_return: float = 0.0,
) -> PortfolioSolution:
    """Minimise annual volatility subject to an expected-return floor."""
    (
        asset_names,
        expected_values,
        covariance_values,
        cholesky_factor,
    ) = _prepare_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
    )

    _validate_scalar(
        target_return,
        name="target_return",
    )
    _validate_scalar(
        cash_expected_return,
        name="cash_expected_return",
    )

    with Model("MinimiseRiskAtReturn") as model:
        weights = model.variable(
            "weights",
            len(asset_names),
            Domain.greaterThan(0.0),
        )
        cash_weight = model.variable(
            "cash_weight",
            Domain.greaterThan(0.0),
        )
        portfolio_risk = model.variable(
            "portfolio_risk",
            Domain.greaterThan(0.0),
        )

        # Risky assets and cash must exhaust the available portfolio capital
        model.constraint(
            "budget",
            Expr.add(
                Expr.sum(weights),
                cash_weight,
            ),
            Domain.equalsTo(1.0),
        )

        portfolio_return = _portfolio_return_expression(
            weights=weights,
            cash_weight=cash_weight,
            expected_returns=expected_values,
            cash_expected_return=cash_expected_return,
        )

        model.constraint(
            "return_floor",
            portfolio_return,
            Domain.greaterThan(target_return),
        )

        # The cone enforces portfolio_risk >= ||G^T x||_2
        model.constraint(
            "risk",
            Expr.vstack(
                portfolio_risk,
                Expr.mul(
                    cholesky_factor.T,
                    weights,
                ),
            ),
            Domain.inQCone(),
        )

        model.objective(
            "portfolio_risk",
            ObjectiveSense.Minimize,
            portfolio_risk,
        )

        model.solve()
        _require_optimal_solution(model)

        return _build_solution(
            asset_names=asset_names,
            weights=np.asarray(
                weights.level(),
                dtype=float,
            ),
            cash_weight=float(cash_weight.level()[0]),
            expected_returns=expected_values,
            covariance=covariance_values,
            cash_expected_return=cash_expected_return,
        )


def generate_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversions: Iterable[float],
    cash_expected_return: float = 0.0,
) -> EfficientFrontierResult:
    """Generate efficient portfolios across a range of risk aversions."""
    (
        asset_names,
        expected_values,
        covariance_values,
        cholesky_factor,
    ) = _prepare_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
    )

    risk_aversion_values = np.asarray(
        list(risk_aversions),
        dtype=float,
    )

    if risk_aversion_values.size == 0:
        raise ValueError("At least one risk-aversion value is required.")

    if not np.isfinite(risk_aversion_values).all():
        raise ValueError("Risk-aversion values must be finite.")

    if (risk_aversion_values < 0.0).any():
        raise ValueError("Risk-aversion values cannot be negative.")

    _validate_scalar(
        cash_expected_return,
        name="cash_expected_return",
    )

    summaries: list[dict[str, float]] = []
    allocations: list[dict[str, float]] = []

    with Model("EfficientFrontier") as model:
        weights = model.variable(
            "weights",
            len(asset_names),
            Domain.greaterThan(0.0),
        )
        cash_weight = model.variable(
            "cash_weight",
            Domain.greaterThan(0.0),
        )
        portfolio_risk = model.variable(
            "portfolio_risk",
            Domain.greaterThan(0.0),
        )

        # A Fusion parameter allows repeated solves without rebuilding the model
        risk_aversion = model.parameter("risk_aversion")

        model.constraint(
            "budget",
            Expr.add(
                Expr.sum(weights),
                cash_weight,
            ),
            Domain.equalsTo(1.0),
        )

        model.constraint(
            "risk",
            Expr.vstack(
                portfolio_risk,
                Expr.mul(
                    cholesky_factor.T,
                    weights,
                ),
            ),
            Domain.inQCone(),
        )

        portfolio_return = _portfolio_return_expression(
            weights=weights,
            cash_weight=cash_weight,
            expected_returns=expected_values,
            cash_expected_return=cash_expected_return,
        )

        # Objective = expected return - risk_aversion * annual volatility
        risk_adjusted_return = Expr.sub(
            portfolio_return,
            Expr.mul(
                risk_aversion,
                portfolio_risk,
            ),
        )

        model.objective(
            "risk_adjusted_return",
            ObjectiveSense.Maximize,
            risk_adjusted_return,
        )

        for value in risk_aversion_values:
            risk_aversion.setValue(float(value))

            model.solve()
            _require_optimal_solution(model)

            solution = _build_solution(
                asset_names=asset_names,
                weights=np.asarray(
                    weights.level(),
                    dtype=float,
                ),
                cash_weight=float(cash_weight.level()[0]),
                expected_returns=expected_values,
                covariance=covariance_values,
                cash_expected_return=cash_expected_return,
            )

            summaries.append(
                {
                    "risk_aversion": float(value),
                    "expected_return": solution.expected_return,
                    "volatility": solution.volatility,
                    "cash_weight": solution.cash_weight,
                }
            )

            allocation = solution.weights.to_dict()
            allocation["Cash"] = solution.cash_weight
            allocations.append(allocation)

    return EfficientFrontierResult(
        summary=pd.DataFrame(summaries),
        weights=pd.DataFrame(allocations),
    )


def _portfolio_return_expression(
    weights: Variable,
    cash_weight: Variable,
    expected_returns: np.ndarray,
    cash_expected_return: float,
) -> Expression:
    """Build the expected-return expression shared by each model."""
    risky_return = Expr.dot(
        expected_returns,
        weights,
    )

    # Cash is modelled separately with zero covariance against risky assets
    cash_return = Expr.mul(
        cash_expected_return,
        cash_weight,
    )

    return Expr.add(
        risky_return,
        cash_return,
    )


def _prepare_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate, align and factorise optimisation inputs."""
    if expected_returns.empty:
        raise ValueError("Expected returns cannot be empty.")

    if covariance.empty:
        raise ValueError("Covariance matrix cannot be empty.")

    if not expected_returns.index.is_unique:
        raise ValueError("Expected-return asset names must be unique.")

    if not covariance.index.is_unique or not covariance.columns.is_unique:
        raise ValueError("Covariance asset names must be unique.")

    if not is_numeric_dtype(expected_returns.dtype):
        raise TypeError("Expected returns must be numeric.")

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

    if set(covariance.index) != asset_set or set(covariance.columns) != asset_set:
        raise ValueError(
            "Expected returns and covariance must contain the same assets."
        )

    # Expected-return ordering defines the vector and matrix ordering in the model
    aligned_covariance = covariance.loc[
        asset_names,
        asset_names,
    ]

    expected_values = expected_returns.to_numpy(dtype=float)
    covariance_values = aligned_covariance.to_numpy(dtype=float)

    if not np.isfinite(expected_values).all():
        raise ValueError("Expected returns must be finite.")

    if not np.isfinite(covariance_values).all():
        raise ValueError("Covariance values must be finite.")

    if not np.allclose(
        covariance_values,
        covariance_values.T,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Covariance matrix must be symmetric.")

    try:
        # Cholesky gives Sigma = GG^T for the conic volatility formulation
        cholesky_factor = np.linalg.cholesky(covariance_values)
    except np.linalg.LinAlgError as error:
        raise ValueError("Covariance matrix must be positive definite.") from error

    return (
        asset_names,
        expected_values,
        covariance_values,
        cholesky_factor,
    )


def _build_solution(
    asset_names: list[str],
    weights: np.ndarray,
    cash_weight: float,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    cash_expected_return: float,
) -> PortfolioSolution:
    """Convert raw solver values into labelled portfolio statistics."""
    portfolio_return = float(
        weights @ expected_returns + cash_weight * cash_expected_return
    )

    # Recalculate volatility from Sigma to verify the realised portfolio risk
    portfolio_variance = float(weights @ covariance @ weights)
    portfolio_volatility = float(np.sqrt(max(portfolio_variance, 0.0)))

    return PortfolioSolution(
        weights=pd.Series(
            weights,
            index=asset_names,
            name="weight",
        ),
        cash_weight=cash_weight,
        expected_return=portfolio_return,
        volatility=portfolio_volatility,
    )


def _require_optimal_solution(model: Model) -> None:
    """Require an optimal primal solution before reading solver values."""
    status = model.getPrimalSolutionStatus()

    if status != SolutionStatus.Optimal:
        raise RuntimeError(f"MOSEK did not return an optimal solution: {status}")


def _validate_scalar(
    value: float,
    name: str,
    minimum: float | None = None,
) -> None:
    """Validate a finite scalar optimisation input."""
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
