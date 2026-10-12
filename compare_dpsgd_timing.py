#!/usr/bin/env python3
"""Compare and time fixed-size replace-one DP-SGD accountants per case.

This script performs the same type of numerical evaluation as
``compare_dpsgd.py``, but every fixed parameter tuple

    (q, noise multiplier, steps, privacy delta)

is benchmarked independently. For each tuple, ``--repeats`` repeats
all three RDP methods from scratch and reports average wall-clock runtime.

For the new proposed RDP bound, first a binary search on
orders from 2 through ``alpha_max`` using SciPy-based calculations is done. This is followed by FLINT-based calculation for
the orders within the window around the selected order.

Time spent for importing modules is excluded.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import gc
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, TypeVar

import numpy as np

from birrell_rdp import fswo_replace_one_rdp_profile as birrell_profile
from isosceles_rdp_scipy import find_optimal_order_binary
from rdp_to_dp import rdp_to_dp_at_order, rdp_to_dp
from wbk_rdp import fswo_replace_one_rdp_profile as wbk_profile


T = TypeVar("T")
BIRRELL_TAYLOR_ORDER = 3

ORIGINAL_FIELDS = [
    "sampling_probability_q",
    "noise_multiplier_sigma",
    "replacement_shift_Delta",
    "steps",
    "nominal_epochs_Tq",
    "privacy_delta",
    "isosceles_epsilon_upper",
    "isosceles_epsilon_upper_text",
    "isosceles_scipy_epsilon_estimate",
    "isosceles_scipy_optimal_alpha",
    "isosceles_scipy_best_u",
    "isosceles_scipy_one_step_estimated_error",
    "isosceles_scipy_all_orders_converged",
    "birrell_epsilon",
    "birrell_optimal_alpha",
    "wbk_epsilon",
    "wbk_optimal_alpha",
    "alpha_max",
    "total_rdp_numerical_error_budget",
    "scipy_requested_accuracy",
    "rdp_conversion",
]

BIRRELL_FIELDS = ["birrell_taylor_order"]

CSV_FIELDS = ORIGINAL_FIELDS + BIRRELL_FIELDS


def _comma_list(text: str, cast: Callable[[str], Any]) -> list[Any]:
    values = [cast(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not values:
        raise ValueError("empty comma-separated list")
    return values


def _positive_fraction(value: str | int | float) -> Fraction:
    result = Fraction(str(value))
    if result <= 0:
        raise ValueError("expected a positive number")
    return result


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _order_window(
    alpha: int,
    radius: int,
    alpha_max: int,
) -> list[int]:
    """Return the order window around ``alpha``."""
    return list(
        range(
            max(2, alpha - radius),
            min(alpha + radius, alpha_max) + 1,
        )
    )


def _table_text(key: str, value: Any) -> str:
    if key == "privacy_delta":
        return format(Decimal(str(value)), "f")
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("sampling_probability_q", "q"),
        ("sigma", "sigma"),
        ("steps", "T"),
        ("privacy_delta", "delta"),
        ("isosceles_epsilon_upper", "iso cert"),
        ("isosceles_scipy_epsilon_estimate", "iso scipy"),
        ("birrell_epsilon", "Birrell"),
        ("birrell_taylor_order", "Birrell m"),
        ("wbk_epsilon", "WBK"),
    ]
    widths = {
        key: max(
            len(label),
            12,
            *(len(_table_text(key, row[key])) for row in rows),
        )
        for key, label in columns
    }

    print("\nFinal epsilon table")
    print("  ".join(label.rjust(widths[key]) for key, label in columns))
    print("  ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print(
            "  ".join(
                _table_text(key, row[key]).rjust(widths[key])
                for key, _ in columns
            )
        )

# Each isosceles timing sample includes one SciPy search, all FLINT work in its
# selected window, and the best final RDP-to-DP conversion.
COMPARISON_METHODS = [
    (
        "isosceles_scipy_binary_search_plus_flint",
        "Isosceles: SciPy binary search + FLINT",
    ),
    ("birrell", "Birrell"),
    ("wbk", "WBK"),
]
METHOD_LABELS = dict(COMPARISON_METHODS)
METHOD_ORDER = {
    method: index
    for index, (method, _) in enumerate(COMPARISON_METHODS)
}

TIMING_COMPARISON_FIELDS = [
    "repeats",
    "timing_scope",
    "isosceles_scipy_binary_search_plus_flint_average_seconds",
    "birrell_average_seconds",
    "wbk_average_seconds",
]
COMPARISON_FIELDS = CSV_FIELDS + TIMING_COMPARISON_FIELDS

TIMING_FIELDS = [
    "scope",
    "timing_category",
    "sampling_probability_q",
    "noise_multiplier_sigma",
    "replacement_shift_Delta",
    "steps",
    "nominal_epochs_Tq",
    "privacy_delta",
    "alpha_max",
    "selected_alpha",
    "method",
    "method_label",
    "repeats",
    "average_seconds",
    "median_seconds",
    "minimum_seconds",
    "maximum_seconds",
    "sample_standard_deviation_seconds",
    "directly_timed",
    "timed_workload",
    "prerequisite",
    "clock",
    "module_import_time_included",
    "pre_timing_gc_included",
    "first_repetition_included",
]

TIMING_SAMPLE_FIELDS = [
    "scope",
    "timing_category",
    "sampling_probability_q",
    "noise_multiplier_sigma",
    "replacement_shift_Delta",
    "steps",
    "nominal_epochs_Tq",
    "privacy_delta",
    "alpha_max",
    "selected_alpha",
    "method",
    "method_label",
    "repetition",
    "elapsed_seconds",
    "directly_timed",
]


# Keep every raw repetition as well as summary statistics.
@dataclass(frozen=True)
class TimingStatistics:
    """Wall-clock samples and their summary statistics."""

    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("at least one timing sample is required")
        if not all(
            math.isfinite(value) and value >= 0.0 for value in self.samples
        ):
            raise ValueError("timing samples must be finite and nonnegative")

    @property
    def repeats(self) -> int:
        return len(self.samples)

    @property
    def average(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def minimum(self) -> float:
        return min(self.samples)

    @property
    def maximum(self) -> float:
        return max(self.samples)

    @property
    def sample_standard_deviation(self) -> float:
        return statistics.stdev(self.samples) if self.repeats >= 2 else 0.0



def _time_repeated(
    method_label: str,
    repeats: int,
    calculation: Callable[[], T],
    *,
    consistency_key: Callable[[T], Any] | None = None,
) -> tuple[T, TimingStatistics]:
    """Run one fixed-case calculation repeatedly and retain the first result."""
    representative: T | None = None
    representative_key: Any = None
    has_representative = False
    elapsed_values: list[float] = []

    # Every repetition invokes calculation() anew. Garbage collection is
    # completed before the timer and lazy imports are preloaded elsewhere. Any
    # cache owned internally by a backend remains subject to that backend's
    # existing behavior, so the raw samples reveal cold-versus-warm differences.
    for repetition in range(1, repeats + 1):
        # Collection is deliberately outside the timed region.
        gc.collect()
        started = time.perf_counter()
        current = calculation()
        elapsed = time.perf_counter() - started
        elapsed_values.append(elapsed)

        # The first result is retained for the privacy comparison row. For
        # methods with a discrete selected order, consistency_key verifies that
        # repeated timing runs did not silently benchmark different workloads.
        if not has_representative:
            representative = current
            has_representative = True
            if consistency_key is not None:
                representative_key = consistency_key(current)
        elif consistency_key is not None:
            current_key = consistency_key(current)
            if current_key != representative_key:
                raise RuntimeError(
                    f"{method_label} changed a workload-defining result across "
                    f"repetitions: {representative_key!r} versus {current_key!r}"
                )

        print(
            f"      repetition {repetition:>{len(str(repeats))}d}/{repeats}: "
            f"{elapsed:.6f}s",
            flush=True,
        )
        del current

    if not has_representative:
        raise RuntimeError("no timing repetitions were executed")

    timing = TimingStatistics(tuple(elapsed_values))
    print(f"      average: {timing.average:.6f}s", flush=True)
    return representative, timing  


def _load_max_renyi() -> Callable[..., Any]:
    """Load the module for FLINT-based calculation before any timer starts."""
    try:
        from gaussian_mixture_max_renyi import max_renyi
    except ImportError as exc:  
        raise ImportError("This requires python-flint.") from exc
    return max_renyi


# Optional backends import lazily inside their adapters. Import them once here
# before timing so one method is not charged for Python module initialization
# while another measures only numerical work.
def _preload_lazy_backends() -> None:
    """Resolve optional backend imports before wall-clock measurement."""
    try:
        from dp_accounting import dp_event, privacy_accountant
        from dp_accounting.rdp import rdp_privacy_accountant
    except ImportError as exc:  
        raise ImportError(
            "Install `dp-accounting` to run the WBK comparison."
        ) from exc

    _ = (
        dp_event,
        privacy_accountant,
        rdp_privacy_accountant,
    )


# One summary row corresponds to one method on one fixed (q,sigma,T,delta)
# tuple. The metadata states exactly what was included in the timed region.
def _timing_row(
    *,
    category: str,
    q: float,
    sigma: float,
    shift: float,
    steps: int,
    target_delta: float,
    alpha_max: int,
    selected_alpha: int | str,
    method: str,
    timing: TimingStatistics,
    directly_timed: bool,
    workload: str,
    prerequisite: str,
) -> dict[str, Any]:
    return {
        "scope": "one_fixed_parameter_case",
        "timing_category": category,
        "sampling_probability_q": q,
        "noise_multiplier_sigma": sigma,
        "replacement_shift_Delta": shift,
        "steps": steps,
        "nominal_epochs_Tq": steps * q,
        "privacy_delta": target_delta,
        "alpha_max": alpha_max,
        "selected_alpha": selected_alpha,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "repeats": timing.repeats,
        "average_seconds": timing.average,
        "median_seconds": timing.median,
        "minimum_seconds": timing.minimum,
        "maximum_seconds": timing.maximum,
        "sample_standard_deviation_seconds": timing.sample_standard_deviation,
        "directly_timed": directly_timed,
        "timed_workload": workload,
        "prerequisite": prerequisite,
        "clock": "time.perf_counter wall-clock seconds",
        "module_import_time_included": False,
        "pre_timing_gc_included": False,
        "first_repetition_included": True,
    }


# Expand the same timing object into one row per repetition.
def _timing_sample_rows(
    *,
    category: str,
    q: float,
    sigma: float,
    shift: float,
    steps: int,
    target_delta: float,
    alpha_max: int,
    selected_alpha: int | str,
    method: str,
    timing: TimingStatistics,
    directly_timed: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "scope": "one_fixed_parameter_case",
            "timing_category": category,
            "sampling_probability_q": q,
            "noise_multiplier_sigma": sigma,
            "replacement_shift_Delta": shift,
            "steps": steps,
            "nominal_epochs_Tq": steps * q,
            "privacy_delta": target_delta,
            "alpha_max": alpha_max,
            "selected_alpha": selected_alpha,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "repetition": repetition,
            "elapsed_seconds": elapsed,
            "directly_timed": directly_timed,
        }
        for repetition, elapsed in enumerate(timing.samples, start=1)
    ]


def _print_timing_table(
    rows: list[dict[str, Any]],
    *,
    title: str,
) -> None:
    print(f"\n{title}")
    headers = (
        ("sampling_probability_q", "q"),
        ("noise_multiplier_sigma", "sigma"),
        ("steps", "T"),
        ("privacy_delta", "delta"),
        ("method_label", "method"),
        ("selected_alpha", "alpha"),
        ("repeats", "repeats"),
        ("average_seconds", "average (s)"),
        ("minimum_seconds", "minimum (s)"),
        ("maximum_seconds", "maximum (s)"),
        ("sample_standard_deviation_seconds", "sample std (s)"),
    )

    def text(key: str, value: Any) -> str:
        if key == "privacy_delta":
            return f"{float(value):.12g}"
        if key in {
            "average_seconds",
            "minimum_seconds",
            "maximum_seconds",
            "sample_standard_deviation_seconds",
        }:
            return f"{float(value):.6f}"
        return str(value)

    widths = {
        key: max(
            len(label),
            *(len(text(key, row[key])) for row in rows),
        )
        for key, label in headers
    }
    print("  ".join(label.rjust(widths[key]) for key, label in headers))
    print("  ".join("-" * widths[key] for key, _ in headers))
    for row in rows:
        print(
            "  ".join(
                text(key, row[key]).rjust(widths[key])
                for key, _ in headers
            )
        )


def run_experiment(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    # Parse exact q and sigma text for FLINT-based calculation while retaining float versions for
    # SciPy and third-party accountants. Every timing loop below fixes all four
    # parameters before any method is repeated.
    q_texts = _comma_list(str(args.q), str)
    q_fractions = [_positive_fraction(text) for text in q_texts]
    if any(q >= 1 for q in q_fractions):
        raise ValueError("q values must lie in (0,1)")
    q_values = [float(q) for q in q_fractions]

    sigma_texts = _comma_list(args.noise_multipliers, str)
    sigma_values = [float(_positive_fraction(text)) for text in sigma_texts]
    steps_values = _comma_list(args.steps, int)
    delta_texts = _comma_list(args.deltas, str)
    delta_values = [float(_positive_fraction(text)) for text in delta_texts]

    if min(steps_values) <= 0:
        raise ValueError("steps must be positive")
    if not all(0.0 < delta < 1.0 for delta in delta_values):
        raise ValueError("privacy deltas must lie in (0,1)")
    if args.alpha_max < 2:
        raise ValueError("alpha-max must be at least 2")
    if args.order_window < 0:
        raise ValueError("order window must be nonnegative")
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")

    # The RDP methods (Wang et al., Birrell et al.) test integer orders 2,...,alpha_max. numerical_error is
    # a final composed RDP error allowance.
    orders = np.arange(2, args.alpha_max + 1, dtype=int)
    numerical_error = _positive_fraction(args.numerical_error)
    scipy_accuracy = float(args.scipy_accuracy)

    # Imports are deliberately completed before any wall-clock timer starts.
    max_renyi = _load_max_renyi()
    _preload_lazy_backends()

    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    timing_sample_rows: list[dict[str, Any]] = []

    # The nesting order makes one timing unit explicit: after q, sigma, T, and
    # delta are fixed, every method is executed repeats times before the
    # code advances to another parameter value.
    for q_text, q_fraction, q in zip(q_texts, q_fractions, q_values):
        q_fraction_text = _fraction_text(q_fraction)
        for sigma_text, sigma in zip(sigma_texts, sigma_values):
            shift = 2.0 / sigma
            shift_fraction = _positive_fraction(2) / _positive_fraction(sigma_text)
            shift_fraction_text = _fraction_text(shift_fraction)

            # Taking gamma=numerical_error/T keeps the total numerical RDP inflation within numerical_error
            # for this exact timing row.
            for steps in steps_values:
                one_step_tolerance = numerical_error / steps
                one_step_tolerance_text = _fraction_text(one_step_tolerance)

                for delta_text, target_delta in zip(delta_texts, delta_values):
                    print(
                        f"\nFixed case: q={q:g}, sigma={sigma:g}, "
                        f"Delta={shift:g}, T={steps}, delta={delta_text}, "
                        f"orders=2,...,{args.alpha_max}, "
                        f"timing repetitions={args.repeats}",
                        flush=True,
                    )

                    # Wang et al. and Birrell et al. timing functions both include construction
                    # of the complete one-step order profile and minimization of
                    # the final improved RDP-to-DP conversion for this tuple.
                    def calculate_wbk() -> tuple[np.ndarray, Any]:
                        profile = wbk_profile(orders, sigma, q_text)
                        conversion = rdp_to_dp(
                            orders,
                            profile,
                            steps,
                            target_delta,
                        )
                        return profile, conversion

                    print("  Timing WBK full order search", flush=True)
                    wbk_output, wbk_timing = _time_repeated(
                        METHOD_LABELS["wbk"],
                        args.repeats,
                        calculate_wbk,
                        consistency_key=lambda output: int(
                            output[1].optimal_alpha
                        ),
                    )
                    wbk_rdp, wbk_conversion = wbk_output
                    # Time one binary search (SciPy-based calulation), plus FLINT-based calculation at Renyi orders in the window,
                    # and the best final epsilon together.
                    def calculate_isosceles_end_to_end() -> tuple[Any, Any]:
                        search = find_optimal_order_binary(
                            q,
                            sigma,
                            steps,
                            target_delta,
                            accuracy=scipy_accuracy,
                            alpha_max=args.alpha_max,
                        )
                        window = _order_window(
                            int(search.optimal_alpha),
                            args.order_window,
                            args.alpha_max,
                        )
                        # The same final RDP error allowance is given to every
                        # alpha. Each conversion is individually valid,
                        # so minimizing across the window remains a valid bound.
                        conversions = []
                        for alpha in window:
                            calculated = max_renyi(
                                p=q_fraction_text,
                                delta=shift_fraction_text,
                                alpha=alpha,
                                gamma=one_step_tolerance_text,
                            )
                            conversions.append(
                                rdp_to_dp_at_order(
                                    alpha,
                                    calculated.upper,
                                    steps,
                                    delta_text,
                                )
                            )
                        return search, min(
                            conversions,
                            key=lambda result: result.epsilon,
                        )

                    print(
                        "  Timing SciPy-based binary search plus "
                        "FLINT-based calculation ",
                        flush=True,
                    )
                    calculated_output, calculated_timing = _time_repeated(
                        METHOD_LABELS["isosceles_scipy_binary_search_plus_flint"],
                        args.repeats,
                        calculate_isosceles_end_to_end,
                        consistency_key=lambda output: (
                            int(output[0].optimal_alpha),
                            int(output[1].optimal_alpha),
                        ),
                    )
                    scipy_search, calculated_conversion = calculated_output
                    scipy_selected_alpha = int(scipy_search.optimal_alpha)
                    boundary = (
                        " (alpha_max boundary)"
                        if scipy_search.hit_upper_boundary
                        else ""
                    )
                    print(
                        f"    selected alpha={scipy_selected_alpha}{boundary}; "
                        f"queried orders {list(scipy_search.evaluated_orders)}",
                        flush=True,
                    )
                    calculated_orders = _order_window(
                        scipy_selected_alpha,
                        args.order_window,
                        args.alpha_max,
                    )

                    def calculate_birrell() -> tuple[np.ndarray, Any]:
                        # Timing uses only m=3.
                        profile = birrell_profile(
                            orders,
                            noise_multiplier=sigma,
                            sampling_probability=q,
                            taylor_order=BIRRELL_TAYLOR_ORDER,
                        )
                        conversion = rdp_to_dp(
                            orders,
                            profile,
                            steps,
                            target_delta,
                        )
                        return profile, conversion

                    print(
                        f"  Timing Birrell m={BIRRELL_TAYLOR_ORDER} full order search",
                        flush=True,
                    )
                    birrell_output, birrell_timing = _time_repeated(
                        METHOD_LABELS["birrell"],
                        args.repeats,
                        calculate_birrell,
                        consistency_key=lambda output: int(
                            output[1].optimal_alpha
                        ),
                    )
                    birrell_rdp, birrell_conversion = birrell_output

                    # Use the representative first repetition from each method
                    # to populate the numerical comparison row; the timing rows
                    # separately retain all repetitions.
                    scipy_detail = scipy_search.optimal_detail
                    all_scipy_orders_converged = (
                        scipy_search.all_evaluated_orders_converged
                    )
                    row = {
                        "sampling_probability_q": q,
                        "noise_multiplier_sigma": sigma,
                        "replacement_shift_Delta": shift,
                        "steps": steps,
                        "nominal_epochs_Tq": steps * q,
                        "privacy_delta": target_delta,
                        "isosceles_epsilon_upper": calculated_conversion.epsilon,
                        "isosceles_epsilon_upper_text": (
                            calculated_conversion.upper_text
                        ),
                        "isosceles_scipy_epsilon_estimate": (
                            scipy_search.epsilon
                        ),
                        "isosceles_scipy_optimal_alpha": scipy_selected_alpha,
                        "isosceles_scipy_best_u": float(
                            scipy_detail.maximizing_u
                        ),
                        "isosceles_scipy_one_step_estimated_error": float(
                            scipy_detail.estimated_error
                        ),
                        "isosceles_scipy_all_orders_converged": (
                            all_scipy_orders_converged
                        ),
                        "birrell_epsilon": birrell_conversion.epsilon,
                        "birrell_optimal_alpha": (
                            birrell_conversion.optimal_alpha
                        ),
                        "birrell_taylor_order": BIRRELL_TAYLOR_ORDER,
                        "wbk_epsilon": wbk_conversion.epsilon,
                        "wbk_optimal_alpha": wbk_conversion.optimal_alpha,
                        "alpha_max": args.alpha_max,
                        "total_rdp_numerical_error_budget": float(
                            numerical_error
                        ),
                        "scipy_requested_accuracy": scipy_accuracy,
                        "rdp_conversion": (
                            "improved standard RDP-to-DP conversion"
                        ),
                        "repeats": args.repeats,
                        "timing_scope": (
                            "one fixed (q, sigma, steps, privacy_delta) tuple per "
                            "timing sample"
                        ),
                        "isosceles_scipy_binary_search_plus_flint_average_seconds": (
                            calculated_timing.average
                        ),
                        "birrell_average_seconds": birrell_timing.average,
                        "wbk_average_seconds": wbk_timing.average,
                    }
                    rows.append(row)

                    # Describe each timed workload explicitly in the long-form
                    # CSV.
                    timing_specs = [
                        (
                            "comparison_method",
                            "isosceles_scipy_binary_search_plus_flint",
                            int(calculated_conversion.optimal_alpha),
                            calculated_timing,
                            True,
                            (
                                "one SciPy-based binary search "
                                f"followed by FLINT-based calculation on window "
                                f"{calculated_orders} centered at "
                                f"alpha={scipy_selected_alpha}"
                            ),
                            "SciPy order selection is included",
                        ),
                        (
                            "comparison_method",
                            "birrell",
                            int(birrell_conversion.optimal_alpha),
                            birrell_timing,
                            True,
                            (
                                f"Taylor order m={BIRRELL_TAYLOR_ORDER}; orders 2 "
                                f"through {args.alpha_max}, followed by one RDP-to-DP "
                                "minimization for this fixed tuple"
                            ),
                            "none",
                        ),
                        (
                            "comparison_method",
                            "wbk",
                            int(wbk_conversion.optimal_alpha),
                            wbk_timing,
                            True,
                            (
                                f"orders 2 through {args.alpha_max}, followed by one "
                                "RDP-to-DP minimization for this fixed tuple"
                            ),
                            "none",
                        ),
                    ]

                    for (
                        category,
                        method,
                        method_alpha,
                        timing,
                        directly_timed,
                        workload,
                        prerequisite,
                    ) in timing_specs:
                        timing_rows.append(
                            _timing_row(
                                category=category,
                                q=q,
                                sigma=sigma,
                                shift=shift,
                                steps=steps,
                                target_delta=target_delta,
                                alpha_max=args.alpha_max,
                                selected_alpha=method_alpha,
                                method=method,
                                timing=timing,
                                directly_timed=directly_timed,
                                workload=workload,
                                prerequisite=prerequisite,
                            )
                        )
                        timing_sample_rows.extend(
                            _timing_sample_rows(
                                category=category,
                                q=q,
                                sigma=sigma,
                                shift=shift,
                                steps=steps,
                                target_delta=target_delta,
                                alpha_max=args.alpha_max,
                                selected_alpha=method_alpha,
                                method=method,
                                timing=timing,
                                directly_timed=directly_timed,
                            )
                        )

    rows.sort(
        key=lambda row: (
            row["sampling_probability_q"],
            row["noise_multiplier_sigma"],
            row["steps"],
            row["privacy_delta"],
        )
    )
    timing_rows.sort(
        key=lambda row: (
            row["sampling_probability_q"],
            row["noise_multiplier_sigma"],
            row["steps"],
            row["privacy_delta"],
            0 if row["timing_category"] == "comparison_method" else 1,
            METHOD_ORDER[row["method"]],
        )
    )
    timing_sample_rows.sort(
        key=lambda row: (
            row["sampling_probability_q"],
            row["noise_multiplier_sigma"],
            row["steps"],
            row["privacy_delta"],
            0 if row["timing_category"] == "comparison_method" else 1,
            METHOD_ORDER[row["method"]],
            row["repetition"],
        )
    )

    # Write three files: privacy results with average times,
    # per-method timing summaries, and every individual wall-clock sample.
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison_path = (
        output_directory
        / "dpsgd_comparison_with_timing.csv"
    )
    timing_path = output_directory / "dpsgd_timings.csv"
    timing_samples_path = (
        output_directory / "dpsgd_timing_samples.csv"
    )

    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with timing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(timing_rows)

    with timing_samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(timing_sample_rows)

    display_rows = [
        {
            "sampling_probability_q": row["sampling_probability_q"],
            "sigma": row["noise_multiplier_sigma"],
            "steps": row["steps"],
            "privacy_delta": row["privacy_delta"],
            **{
                key: row[key]
                for key in (
                    "isosceles_epsilon_upper",
                    "isosceles_scipy_epsilon_estimate",
                    "birrell_epsilon",
                    "birrell_taylor_order",
                    "wbk_epsilon",
                )
            },
        }
        for row in rows
    ]
    _print_table(display_rows)

    comparison_timing_rows = [
        row
        for row in timing_rows
        if row["timing_category"] == "comparison_method"
    ]
    _print_timing_table(
        comparison_timing_rows,
        title="Per-fixed-parameter runtime for the three RDP methods",
    )
    print(f"\nWrote {comparison_path}")
    print(f"Wrote {timing_path}")
    print(f"Wrote {timing_samples_path}")
    return rows, timing_rows, timing_sample_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q",
        default="0.01",
        help="batch fraction b/n; comma-separated values are allowed",
    )
    parser.add_argument("--noise-multipliers", default="2")
    parser.add_argument("--steps", default="10000")
    parser.add_argument("--deltas", default="1e-6")
    parser.add_argument(
        "--alpha-max",
        type=int,
        default=64,
        help="largest integer Renyi order",
    )
    parser.add_argument(
        "--numerical-error",
        default="1e-5",
        help="maximum additive numerical RDP inflation after composition for each T",
    )
    parser.add_argument(
        "--order-window",
        type=int,
        default=0,
        help=(
            "number of integer orders on each side of the SciPy-selected order"
        ),
    )
    parser.add_argument(
        "--scipy-accuracy",
        type=float,
        default=1e-6,
        help=(
            "successive-refinement target for each low-accuracy SciPy order "
            "queried by the binary search"
        ),
    )
    parser.add_argument("--output-directory", default="results")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help=(
            "number of timed executions of every method for every fixed "
            "(q, sigma, steps, delta) tuple (default: 5)"
        ),
    )
    return parser


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
