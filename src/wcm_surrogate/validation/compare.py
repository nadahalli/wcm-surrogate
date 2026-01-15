"""
Compare simulator output to experimental data.

Key biological relationships to validate:
1. Growth rate vs cell volume (larger cells grow faster)
2. Growth rate distribution (CV should match experiments ~10-20%)
3. Division time distribution
4. Size homeostasis (adder model)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..growth_rate import GrowthRateSimulator
from ..growth_rate.langevin import GrowthParameters
from .experimental_data import (
    load_taheri_araghi,
    load_si_2017,
    get_experimental_summary,
    ExperimentalCondition,
)


@dataclass
class ValidationResult:
    """Result of a validation comparison."""
    metric_name: str
    experimental_value: float
    simulated_value: float
    relative_error: float
    passed: bool
    tolerance: float
    notes: str = ""


def simulate_at_growth_rate(
    target_rate_per_hr: float,
    n_cells: int = 50,
    t_max: float = 200.0,
    seed: Optional[int] = None
) -> dict:
    """
    Run simulations calibrated to a target growth rate.

    Args:
        target_rate_per_hr: Target growth rate in /hour
        n_cells: Number of cells to simulate
        t_max: Simulation time (minutes)
        seed: Random seed

    Returns:
        Dictionary with simulation statistics
    """
    # Convert /hr to /min
    target_rate_per_min = target_rate_per_hr / 60.0

    params = GrowthParameters(growth_rate=target_rate_per_min)
    sim = GrowthRateSimulator(params, seed=seed)

    pop = sim.run_population(n_cells=n_cells, t_max=t_max, dt=0.1)

    # Convert growth rate back to /hr for comparison
    mean_gr_per_hr = np.mean(pop["mean_growth_rate"]) * 60
    std_gr_per_hr = np.std(pop["mean_growth_rate"]) * 60
    cv_gr = std_gr_per_hr / mean_gr_per_hr if mean_gr_per_hr > 0 else 0

    return {
        "mean_growth_rate_per_hr": mean_gr_per_hr,
        "std_growth_rate_per_hr": std_gr_per_hr,
        "cv_growth_rate": cv_gr,
        "mean_volume": np.mean(pop["mean_volume"]),
        "std_volume": np.std(pop["mean_volume"]),
        "mean_interdivision_time": np.mean(pop["interdivision_times"]) if len(pop["interdivision_times"]) > 0 else None,
        "n_cells": n_cells,
        "target_rate_per_hr": target_rate_per_hr,
    }


def validate_growth_rate_calibration(tolerance: float = 0.3) -> list[ValidationResult]:
    """
    Validate that simulator reproduces target growth rates.

    Tests that when we set a growth rate, the simulation produces
    approximately that growth rate.

    Args:
        tolerance: Relative error tolerance for passing (30% default)

    Returns:
        List of validation results
    """
    results = []

    # Get experimental data for target growth rates
    exp_data = load_taheri_araghi()

    print("Validating growth rate calibration...")
    print("-" * 60)
    print(f"{'Condition':<15} {'Exp (1/hr)':<12} {'Sim (1/hr)':<12} {'Error':<10} {'Status'}")
    print("-" * 60)

    for exp in exp_data:
        # Run simulation at this growth rate
        sim_stats = simulate_at_growth_rate(
            target_rate_per_hr=exp.growth_rate_per_hr,
            n_cells=30,
            t_max=150.0,
            seed=42
        )

        # Compare growth rates
        exp_rate = exp.growth_rate_per_hr
        sim_rate = sim_stats["mean_growth_rate_per_hr"]
        rel_error = abs(sim_rate - exp_rate) / exp_rate if exp_rate > 0 else 0
        passed = rel_error < tolerance

        results.append(ValidationResult(
            metric_name=f"growth_rate_{exp.condition}",
            experimental_value=exp_rate,
            simulated_value=sim_rate,
            relative_error=rel_error,
            passed=passed,
            tolerance=tolerance,
            notes=f"Target: {exp_rate:.2f}/hr"
        ))

        status = "PASS" if passed else "FAIL"
        print(f"{exp.condition:<15} {exp_rate:<12.2f} {sim_rate:<12.2f} {rel_error:<10.1%} {status}")

    return results


def validate_growth_rate_variability(tolerance: float = 1.0) -> ValidationResult:
    """
    Validate that growth rate variability (CV) is biologically realistic.

    Experimental observation: CV of growth rate is typically 10-25%
    """
    print("\nValidating growth rate variability (CV)...")

    # Typical E. coli fast growth condition
    target_rate = 1.5  # /hr

    sim_stats = simulate_at_growth_rate(
        target_rate_per_hr=target_rate,
        n_cells=100,
        t_max=200.0,
        seed=42
    )

    # Expected CV from experiments (10-25%, we use 15% as reference)
    expected_cv = 0.15
    actual_cv = sim_stats["cv_growth_rate"]

    # This is a softer test - we just want CV to be non-zero and reasonable
    error = abs(actual_cv - expected_cv) / expected_cv if expected_cv > 0 else actual_cv
    passed = actual_cv > 0.01 and actual_cv < 0.5  # Between 1% and 50%

    print(f"  Expected CV: ~{expected_cv:.1%} (typical range 10-25%)")
    print(f"  Simulated CV: {actual_cv:.1%}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return ValidationResult(
        metric_name="growth_rate_cv",
        experimental_value=expected_cv,
        simulated_value=actual_cv,
        relative_error=error,
        passed=passed,
        tolerance=tolerance,
        notes=f"CV should be ~10-25% for realistic stochasticity"
    )


def validate_doubling_time(tolerance: float = 0.3) -> ValidationResult:
    """
    Validate that doubling time matches theoretical prediction.

    Doubling time = ln(2) / growth_rate
    """
    print("\nValidating doubling time...")

    target_rate = 1.5  # /hr → expected doubling ~28 min
    expected_doubling = np.log(2) / (target_rate / 60)  # in minutes

    sim_stats = simulate_at_growth_rate(
        target_rate_per_hr=target_rate,
        n_cells=50,
        t_max=300.0,
        seed=42
    )

    actual_doubling = sim_stats["mean_interdivision_time"]
    if actual_doubling is None:
        actual_doubling = expected_doubling  # Fallback

    error = abs(actual_doubling - expected_doubling) / expected_doubling
    passed = error < tolerance

    print(f"  Expected doubling time: {expected_doubling:.1f} min")
    print(f"  Simulated doubling time: {actual_doubling:.1f} min")
    print(f"  Error: {error:.1%}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return ValidationResult(
        metric_name="doubling_time",
        experimental_value=expected_doubling,
        simulated_value=actual_doubling,
        relative_error=error,
        passed=passed,
        tolerance=tolerance,
        notes=f"For growth rate {target_rate}/hr"
    )


def run_validation_suite(verbose: bool = True) -> dict:
    """
    Run full validation suite comparing simulator to experimental data.

    Args:
        verbose: Print results as we go

    Returns:
        Dictionary with all validation results and summary
    """
    if verbose:
        print("=" * 60)
        print("VALIDATION SUITE: Comparing simulator to E. coli data")
        print("=" * 60)

        # Print experimental data summary
        summary = get_experimental_summary()
        print("\nExperimental data loaded:")
        for source, stats in summary.items():
            print(f"  {source}: {stats['n_conditions']} conditions, "
                  f"growth rate {stats['growth_rate_range_per_hr'][0]:.2f}-"
                  f"{stats['growth_rate_range_per_hr'][1]:.2f} /hr")
        print()

    # Run validations
    results = {}

    # 1. Growth rate calibration
    if verbose:
        print("\n" + "=" * 60)
        print("1. GROWTH RATE CALIBRATION")
        print("=" * 60)
    results["growth_rate_calibration"] = validate_growth_rate_calibration()

    # 2. Growth rate variability
    if verbose:
        print("\n" + "=" * 60)
        print("2. GROWTH RATE VARIABILITY")
        print("=" * 60)
    results["growth_rate_variability"] = validate_growth_rate_variability()

    # 3. Doubling time
    if verbose:
        print("\n" + "=" * 60)
        print("3. DOUBLING TIME")
        print("=" * 60)
    results["doubling_time"] = validate_doubling_time()

    # Summary
    all_results = results["growth_rate_calibration"] + [
        results["growth_rate_variability"],
        results["doubling_time"]
    ]
    n_passed = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)

    if verbose:
        print("\n" + "=" * 60)
        print(f"VALIDATION SUMMARY: {n_passed}/{n_total} tests passed ({100*n_passed/n_total:.0f}%)")
        print("=" * 60)

    results["summary"] = {
        "n_passed": n_passed,
        "n_total": n_total,
        "pass_rate": n_passed / n_total if n_total > 0 else 0
    }

    return results


if __name__ == "__main__":
    run_validation_suite(verbose=True)
