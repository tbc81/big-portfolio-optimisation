"""Construct valid correlation stresses and measure portfolio sensitivity."""

from dataclasses import dataclass
from itertools import combinations, pairwise

import numpy as np
import pandas as pd
from mosek.fusion import (
    Domain,
    Expr,
    Matrix,
    Model,
    ObjectiveSense,
    SolutionStatus,
)
from pandas.api.types import is_numeric_dtype

from big_portfolio.analysis.diagnostics import calculate_portfolio_snapshot
from big_portfolio.optimisation.mean_variance import (
    PortfolioSolution,
    minimise_risk_at_return,
)

MINIMUM_CORRELATION_EIGENVALUE = 1e-6
MATRIX_TOLERANCE = 1e-8
SOLVER_TOLERANCE = 1e-7


@dataclass(frozen=True)
class CorrelationStressScenario:
    """Store one correlation-stress scenario and its portfolio impact."""

    label: str
    target_correlation: float | None
    realised_group_correlation: float
    correlation: pd.DataFrame
    covariance: pd.DataFrame
    current_volatility: float
    current_variance_multiplier: float
    efficient_solution: PortfolioSolution
    matrix_distance: float


@dataclass(frozen=True)
class CorrelationStressResult:
    """Store the complete correlation-stress analysis."""

    base_correlation: pd.DataFrame
    scenarios: tuple[CorrelationStressScenario, ...]
    summary: pd.DataFrame
    efficient_weights: pd.DataFrame


def run_correlation_stress(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
    current_cash_weight: float,
    groups: dict[str, list[str]],
    target_correlations: list[float] | tuple[float, ...],
    cash_expected_return: float = 0.0,
) -> CorrelationStressResult:
    """Stress selected correlations and measure portfolio consequences."""
    _validate_stress_groups(
        groups=groups,
        asset_names=expected_returns.index.tolist(),
    )

    target_values = tuple(float(value) for value in target_correlations)
    _validate_target_correlations(target_values)

    base_correlation = covariance_to_correlation(covariance)

    base_snapshot = calculate_portfolio_snapshot(
        expected_returns=expected_returns,
        covariance=covariance,
        weights=current_weights,
        cash_weight=current_cash_weight,
        cash_expected_return=cash_expected_return,
    )

    if base_snapshot.volatility <= 0.0:
        raise ValueError(
            "Base portfolio volatility must be positive for stress comparison."
        )

    # Compare efficient portfolios at the original portfolio expected return
    base_efficient = minimise_risk_at_return(
        expected_returns=expected_returns,
        covariance=covariance,
        target_return=base_snapshot.expected_return,
        cash_expected_return=cash_expected_return,
    )

    base_group_correlation = average_within_group_correlation(
        correlation=base_correlation,
        groups=groups,
    )

    base_scenario = CorrelationStressScenario(
        label="Base",
        target_correlation=None,
        realised_group_correlation=base_group_correlation,
        correlation=base_correlation,
        covariance=covariance.copy(),
        current_volatility=base_snapshot.volatility,
        current_variance_multiplier=1.0,
        efficient_solution=base_efficient,
        matrix_distance=0.0,
    )

    scenarios = [base_scenario]

    for target in target_values:
        stressed_correlation = solve_stressed_correlation(
            base_correlation=base_correlation,
            groups=groups,
            target_correlation=target,
        )

        stressed_covariance = correlation_to_covariance(
            correlation=stressed_correlation,
            base_covariance=covariance,
        )

        stressed_snapshot = calculate_portfolio_snapshot(
            expected_returns=expected_returns,
            covariance=stressed_covariance,
            weights=current_weights,
            cash_weight=current_cash_weight,
            cash_expected_return=cash_expected_return,
        )

        stressed_efficient = minimise_risk_at_return(
            expected_returns=expected_returns,
            covariance=stressed_covariance,
            target_return=base_snapshot.expected_return,
            cash_expected_return=cash_expected_return,
        )

        # Variance multiplier = stressed variance / base variance
        variance_multiplier = (
            stressed_snapshot.volatility / base_snapshot.volatility
        ) ** 2

        matrix_distance = float(
            np.linalg.norm(
                stressed_correlation.to_numpy(dtype=float)
                - base_correlation.to_numpy(dtype=float),
                ord="fro",
            )
        )

        realised_correlation = average_within_group_correlation(
            correlation=stressed_correlation,
            groups=groups,
        )

        scenarios.append(
            CorrelationStressScenario(
                label=_format_stress_label(target),
                target_correlation=target,
                realised_group_correlation=realised_correlation,
                correlation=stressed_correlation,
                covariance=stressed_covariance,
                current_volatility=stressed_snapshot.volatility,
                current_variance_multiplier=float(variance_multiplier),
                efficient_solution=stressed_efficient,
                matrix_distance=matrix_distance,
            )
        )

    scenario_tuple = tuple(scenarios)

    return CorrelationStressResult(
        base_correlation=base_correlation,
        scenarios=scenario_tuple,
        summary=_build_stress_summary(
            scenarios=scenario_tuple,
            base_volatility=base_snapshot.volatility,
        ),
        efficient_weights=_build_weight_table(
            scenarios=scenario_tuple,
        ),
    )


def solve_stressed_correlation(
    base_correlation: pd.DataFrame,
    groups: dict[str, list[str]],
    target_correlation: float,
) -> pd.DataFrame:
    """Find the nearest valid correlation matrix satisfying the stress."""
    _validate_correlation_matrix(base_correlation)
    _validate_target_correlation(target_correlation)
    _validate_stress_groups(
        groups=groups,
        asset_names=base_correlation.index.tolist(),
    )

    asset_names = base_correlation.index.tolist()
    asset_index = {name: index for index, name in enumerate(asset_names)}
    stress_pairs = _collect_stress_pairs(
        groups=groups,
        asset_names=asset_names,
    )

    base_values = base_correlation.to_numpy(dtype=float)
    dimension = len(asset_names)

    with Model("CorrelationStress") as model:
        # PSD matrix variable enforces symmetry and non-negative eigenvalues
        stressed = model.variable(
            "stressed_correlation",
            Domain.inPSDCone(dimension),
        )
        distance = model.variable(
            "distance",
            Domain.greaterThan(0.0),
        )

        # A valid correlation matrix has ones along its diagonal
        model.constraint(
            "unit_diagonal",
            stressed.diag(),
            Domain.equalsTo(1.0),
        )

        # X - epsilon*I >= 0 gives every eigenvalue a small positive floor
        model.constraint(
            "eigenvalue_floor",
            Expr.sub(
                stressed,
                Matrix.diag(
                    dimension,
                    MINIMUM_CORRELATION_EIGENVALUE,
                ),
            ),
            Domain.inPSDCone(),
        )

        for pair_number, (first_asset, second_asset) in enumerate(stress_pairs):
            model.constraint(
                f"pair_floor_{pair_number}",
                stressed.index(
                    asset_index[first_asset],
                    asset_index[second_asset],
                ),
                Domain.greaterThan(target_correlation),
            )

        base_matrix = Matrix.dense(base_values.tolist())
        deviation = Expr.sub(
            stressed,
            base_matrix,
        )

        # ||C* - C||_F is the Euclidean norm of the flattened matrix difference
        model.constraint(
            "matrix_distance",
            Expr.vstack(
                distance,
                Expr.flatten(deviation),
            ),
            Domain.inQCone(),
        )

        model.objective(
            "nearest_stressed_correlation",
            ObjectiveSense.Minimize,
            distance,
        )

        model.solve()
        _require_optimal_stress_solution(model)

        stressed_values = np.asarray(
            stressed.level(),
            dtype=float,
        ).reshape(
            dimension,
            dimension,
        )

    # Remove insignificant solver asymmetry before returning labelled data
    stressed_values = 0.5 * (stressed_values + stressed_values.T)
    np.fill_diagonal(
        stressed_values,
        1.0,
    )

    stressed_correlation = pd.DataFrame(
        stressed_values,
        index=asset_names,
        columns=asset_names,
    )

    _validate_correlation_matrix(stressed_correlation)
    _validate_realised_stress(
        correlation=stressed_correlation,
        stress_pairs=stress_pairs,
        target_correlation=target_correlation,
    )

    return stressed_correlation


def covariance_to_correlation(
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Convert covariance into a labelled correlation matrix."""
    covariance_values = _validate_covariance(covariance)

    volatilities = np.sqrt(np.diag(covariance_values))

    # C_ij = Sigma_ij / (sigma_i * sigma_j)
    correlation_values = covariance_values / np.outer(
        volatilities,
        volatilities,
    )

    # Remove floating-point excursions outside the theoretical correlation bounds
    correlation_values = np.clip(
        correlation_values,
        -1.0,
        1.0,
    )
    np.fill_diagonal(
        correlation_values,
        1.0,
    )

    correlation = pd.DataFrame(
        correlation_values,
        index=covariance.index,
        columns=covariance.columns,
    )

    _validate_correlation_matrix(correlation)

    return correlation


def correlation_to_covariance(
    correlation: pd.DataFrame,
    base_covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild covariance while preserving base asset volatilities."""
    _validate_correlation_matrix(correlation)
    base_covariance_values = _validate_covariance(base_covariance)

    asset_names = base_covariance.index.tolist()

    if set(correlation.index) != set(asset_names) or set(correlation.columns) != set(
        asset_names
    ):
        raise ValueError("Correlation and covariance must contain the same assets.")

    aligned_correlation = correlation.loc[
        asset_names,
        asset_names,
    ]

    volatilities = np.sqrt(np.diag(base_covariance_values))

    # Sigma* = D C* D preserves each asset's original marginal volatility
    covariance_values = aligned_correlation.to_numpy(dtype=float) * np.outer(
        volatilities,
        volatilities,
    )

    covariance = pd.DataFrame(
        covariance_values,
        index=asset_names,
        columns=asset_names,
    )

    _validate_covariance(covariance)

    return covariance


def average_within_group_correlation(
    correlation: pd.DataFrame,
    groups: dict[str, list[str]],
) -> float:
    """Calculate the average correlation across unique selected group pairs."""
    _validate_correlation_matrix(correlation)
    _validate_stress_groups(
        groups=groups,
        asset_names=correlation.index.tolist(),
    )

    stress_pairs = _collect_stress_pairs(
        groups=groups,
        asset_names=correlation.index.tolist(),
    )

    values = [
        float(
            correlation.loc[
                first_asset,
                second_asset,
            ]
        )
        for first_asset, second_asset in stress_pairs
    ]

    return float(np.mean(values))


def build_stress_comparison(
    results: dict[str, CorrelationStressResult],
) -> pd.DataFrame:
    """Combine stress results from multiple portfolio snapshots."""
    if not results:
        raise ValueError("At least one stress result is required.")

    frames: list[pd.DataFrame] = []

    for portfolio_name, result in results.items():
        summary = result.summary.copy()

        summary.insert(
            0,
            "portfolio",
            portfolio_name,
        )

        frames.append(summary)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def calculate_group_exposure(
    weights: pd.Series,
    groups: dict[str, list[str]],
) -> pd.Series:
    """Calculate portfolio weight allocated to each risk group."""
    if weights.empty:
        raise ValueError("Portfolio weights cannot be empty.")

    if not weights.index.is_unique:
        raise ValueError("Portfolio weight labels must be unique.")

    if not is_numeric_dtype(weights.dtype):
        raise TypeError("Portfolio weights must be numeric.")

    weight_values = weights.to_numpy(dtype=float)

    if not np.isfinite(weight_values).all():
        raise ValueError("Portfolio weights must be finite.")

    if not groups:
        raise ValueError("At least one risk group is required.")

    exposures: dict[str, float] = {}

    for group_name, group_assets in groups.items():
        if len(group_assets) != len(set(group_assets)):
            raise ValueError(f"Risk group '{group_name}' contains duplicate assets.")

        active_assets = [asset for asset in group_assets if asset in weights.index]

        exposures[group_name] = float(weights.loc[active_assets].sum())

    return pd.Series(
        exposures,
        name="group_exposure",
        dtype=float,
    )


def _collect_stress_pairs(
    groups: dict[str, list[str]],
    asset_names: list[str],
) -> tuple[tuple[str, str], ...]:
    """Return unique asset pairs selected by the configured stress groups."""
    asset_order = {asset_name: index for index, asset_name in enumerate(asset_names)}

    pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for group_assets in groups.values():
        for first_asset, second_asset in combinations(
            group_assets,
            2,
        ):
            if asset_order[first_asset] <= asset_order[second_asset]:
                pair = (
                    first_asset,
                    second_asset,
                )
            else:
                pair = (
                    second_asset,
                    first_asset,
                )

            # Overlapping groups may select the same pair more than once
            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)
            pairs.append(pair)

    return tuple(pairs)


def _build_stress_summary(
    scenarios: tuple[CorrelationStressScenario, ...],
    base_volatility: float,
) -> pd.DataFrame:
    """Build a compact table of stress-test results."""
    rows: list[dict[str, float | str | None]] = []

    for scenario in scenarios:
        rows.append(
            {
                "scenario": scenario.label,
                "target_correlation": scenario.target_correlation,
                "realised_group_correlation": (scenario.realised_group_correlation),
                "current_volatility": scenario.current_volatility,
                "volatility_change": (scenario.current_volatility - base_volatility),
                "variance_multiplier": (scenario.current_variance_multiplier),
                "efficient_volatility": (scenario.efficient_solution.volatility),
                "matrix_distance": scenario.matrix_distance,
            }
        )

    return pd.DataFrame(rows)


def _build_weight_table(
    scenarios: tuple[CorrelationStressScenario, ...],
) -> pd.DataFrame:
    """Collect same-return efficient weights across stress scenarios."""
    rows: list[dict[str, float | str]] = []

    for scenario in scenarios:
        weights = scenario.efficient_solution.weights.to_dict()

        weights["Cash"] = scenario.efficient_solution.cash_weight
        weights["scenario"] = scenario.label

        rows.append(weights)

    return pd.DataFrame(rows).set_index("scenario")


def _format_stress_label(
    target_correlation: float,
) -> str:
    """Format a correlation floor as a compact percentage label."""
    percentage = f"{100.0 * target_correlation:.2f}".rstrip("0").rstrip(".")

    return f"{percentage}%"


def _validate_target_correlations(
    target_correlations: tuple[float, ...],
) -> None:
    """Validate an ordered set of correlation stress floors."""
    if not target_correlations:
        raise ValueError("At least one target correlation is required.")

    for target_correlation in target_correlations:
        _validate_target_correlation(target_correlation)

    if any(
        current >= following for current, following in pairwise(target_correlations)
    ):
        raise ValueError("Target correlations must be strictly increasing.")


def _validate_target_correlation(
    target_correlation: float,
) -> None:
    """Validate one correlation stress floor."""
    if not np.isfinite(target_correlation):
        raise ValueError("target_correlation must be finite.")

    if not (-1.0 <= target_correlation < 1.0):
        raise ValueError("target_correlation must lie in the interval [-1, 1).")


def _validate_stress_groups(
    groups: dict[str, list[str]],
    asset_names: list[str],
) -> None:
    """Validate stress-group membership against an asset universe."""
    if not groups:
        raise ValueError("At least one stress group is required.")

    if len(asset_names) != len(set(asset_names)):
        raise ValueError("Asset names must be unique.")

    valid_assets = set(asset_names)

    for group_name, group_assets in groups.items():
        if len(group_assets) < 2:
            raise ValueError(
                f"Stress group '{group_name}' must contain at least two assets."
            )

        if len(group_assets) != len(set(group_assets)):
            raise ValueError(f"Stress group '{group_name}' contains duplicate assets.")

        unknown_assets = set(group_assets) - valid_assets

        if unknown_assets:
            raise ValueError(
                f"Stress group '{group_name}' contains unknown assets: "
                f"{sorted(unknown_assets)}"
            )


def _validate_covariance(
    covariance: pd.DataFrame,
) -> np.ndarray:
    """Validate covariance and return its numerical matrix."""
    values = _validate_labelled_matrix(
        matrix=covariance,
        name="Covariance",
    )

    if (np.diag(values) <= 0.0).any():
        raise ValueError("Asset variances must be positive.")

    minimum_eigenvalue = float(np.linalg.eigvalsh(values).min())

    # Stress reoptimisation requires a covariance suitable for Cholesky
    if minimum_eigenvalue <= 0.0:
        raise ValueError("Covariance matrix must be positive definite.")

    return values


def _validate_correlation_matrix(
    correlation: pd.DataFrame,
) -> np.ndarray:
    """Validate a labelled correlation matrix."""
    values = _validate_labelled_matrix(
        matrix=correlation,
        name="Correlation",
    )

    if not np.allclose(
        np.diag(values),
        1.0,
        rtol=0.0,
        atol=MATRIX_TOLERANCE,
    ):
        raise ValueError("Correlation diagonal must equal one.")

    if ((values < -1.0 - MATRIX_TOLERANCE) | (values > 1.0 + MATRIX_TOLERANCE)).any():
        raise ValueError("Correlation values must lie between -1 and 1.")

    minimum_eigenvalue = float(np.linalg.eigvalsh(values).min())

    if minimum_eigenvalue < -MATRIX_TOLERANCE:
        raise ValueError("Correlation matrix must be positive semidefinite.")

    return values


def _validate_labelled_matrix(
    matrix: pd.DataFrame,
    name: str,
) -> np.ndarray:
    """Validate shared structural properties of a labelled symmetric matrix."""
    if matrix.empty:
        raise ValueError(f"{name} matrix cannot be empty.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} matrix must be square.")

    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError(f"{name} matrix labels must be unique.")

    if matrix.index.tolist() != matrix.columns.tolist():
        raise ValueError(f"{name} matrix labels must align.")

    non_numeric_columns = [
        column
        for column in matrix.columns
        if not is_numeric_dtype(matrix[column].dtype)
    ]

    if non_numeric_columns:
        raise TypeError(
            f"{name} matrix must be numeric for: {sorted(non_numeric_columns)}"
        )

    values = matrix.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"{name} matrix values must be finite.")

    if not np.allclose(
        values,
        values.T,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(f"{name} matrix must be symmetric.")

    return values


def _validate_realised_stress(
    correlation: pd.DataFrame,
    stress_pairs: tuple[tuple[str, str], ...],
    target_correlation: float,
) -> None:
    """Verify the solved matrix satisfies every requested correlation floor."""
    for first_asset, second_asset in stress_pairs:
        realised = float(
            correlation.loc[
                first_asset,
                second_asset,
            ]
        )

        if realised < target_correlation - SOLVER_TOLERANCE:
            raise RuntimeError(
                "Solved correlation matrix does not satisfy the requested stress floor."
            )


def _require_optimal_stress_solution(
    model: Model,
) -> None:
    """Require an optimal primal solution before reading solver values."""
    status = model.getPrimalSolutionStatus()

    if status != SolutionStatus.Optimal:
        raise RuntimeError(
            f"MOSEK did not return an optimal correlation-stress solution: {status}"
        )
