#!/usr/bin/env python3
"""Compare privacy guarantees for fixed-size, replace-one DP-SGD.

For each (q, sigma, steps, delta) tuple, SciPy independently selects an integer
Renyi order by binary search. A FLINT-based calculation then independently does the same calculation again, 
from which the obtained epsilon is used.

The comparison includes: the isosceles accountant, Birrell et al., Wang et al., and the Connect-the-Dots PLD accountant.
"""

from __future__ import annotations

import argparse
import csv
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from birrell_rdp import fswo_replace_one_rdp_profile as birrell_profile
from connect_the_dots_accountant import connect_the_dots_epsilon_upper_bounds
from isosceles_rdp_scipy import find_optimal_order_binary
from rdp_to_dp import rdp_to_dp_at_order, rdp_to_dp
from wbk_rdp import fswo_replace_one_rdp_profile as wbk_profile


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

CONNECT_THE_DOTS_FIELDS = ["connect_the_dots_epsilon_upper"]

CSV_FIELDS = ORIGINAL_FIELDS + BIRRELL_FIELDS + CONNECT_THE_DOTS_FIELDS


# Command-line grids are comma-separated. Parsing them once here makes the
# Cartesian product of q, sigma, T, and delta explicit in run_experiment.
def _comma_list(text: str, cast: Callable[[str], Any]) -> list[Any]:
    values = [cast(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not values:
        raise ValueError("empty comma-separated list")
    return values


# Exact Fractions are retained for q, sigma, Delta=2/sigma, and the numerical
# error budget so the FLINT-based calculation does not inherit binary-float input error.
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
    """Return the integer window around ``alpha``."""
    return list(
        range(
            max(2, alpha - radius),
            min(alpha + radius, alpha_max) + 1,
        )
    )


def _calculate_orders(
    q_text: str,
    sigma_text: str,
    orders: list[int],
    one_step_tolerance: Fraction,
) -> dict[int, Any]:
    """calculate the requested one-step RDP orders once each."""
    try:
        from gaussian_mixture_max_renyi import (
            max_renyi,
        )
    except ImportError as exc:
        raise ImportError("The calculation requires python-flint.") from exc

    # Convert the original DP-SGD parameters to the Gaussian-mixture parameters
    # p=q and Delta=2/sigma.
    q = _positive_fraction(q_text)
    if q >= 1:
        raise ValueError("q must lie in (0,1)")
    shift = Fraction(2) / _positive_fraction(sigma_text)

    # No cache is shared across calls to _calculate_orders or across parameter
    # tuples. The dictionary only prevents a duplicate lookup within the
    # current explicitly supplied order list.
    output: dict[int, Any] = {}
    for alpha in orders:
        started = time.perf_counter()
        result = max_renyi(
            p=_fraction_text(q),
            delta=_fraction_text(shift),
            alpha=alpha,
            gamma=_fraction_text(one_step_tolerance),
        )
        output[alpha] = result.upper
        elapsed = time.perf_counter() - started
        shortcut = (
            "; proven u=0 endpoint shortcut"
            if result.used_u_zero_shortcut
            else ""
        )
        print(
            f"    alpha={alpha:3d}: calculated{shortcut}; {elapsed:.2f}s",
            flush=True,
        )
    return output


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
        ("connect_the_dots_epsilon_upper", "CTD upper"),
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


def _finite_float(value: Any, label: str) -> float:
    """Return ``value`` as a finite float or raise a descriptive error."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _validate_ctd_result(value: Any) -> None:
    _finite_float(value, "Connect-the-Dots epsilon")


def _run_with_case_fallback(
    method_name: str,
    cases: list[tuple[int, float]],
    batch_calculation: Callable[[], Any],
    single_calculation: Callable[[int, float], Any],
    extract_case: Callable[[Any, int, float], Any],
    validate: Callable[[Any], None],
) -> tuple[dict[tuple[int, float], Any], dict[tuple[int, float], str]]:
    """Run a lossless accountant, isolating failures to individual cases.

    The original batched calculation is attempted first. If it fails, or if a
    returned value for one case is missing or nonfinite, only the affected
    cases are retried using singleton ``steps`` and ``deltas`` lists.
    """
    successful: dict[tuple[int, float], Any] = {}
    errors: dict[tuple[int, float], str] = {}
    retry: list[tuple[int, float]] = []

    try:
        batch_output = batch_calculation()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ImportError) or len(cases) == 1:
            for case in cases:
                errors[case] = message
            print(
                f"    {method_name} -> error ({message})",
                flush=True,
            )
            return successful, errors
        print(
            f"    {method_name} batch failed: {message}; "
            "retrying each (T, delta) separately",
            flush=True,
        )
        retry.extend(cases)
    else:
        for steps, delta in cases:
            try:
                value = extract_case(batch_output, steps, delta)
                validate(value)
            except Exception:
                retry.append((steps, delta))
            else:
                successful[(steps, delta)] = value

    for steps, delta in retry:
        try:
            single_output = single_calculation(steps, delta)
            value = extract_case(single_output, steps, delta)
            validate(value)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors[(steps, delta)] = message
            print(
                f"    {method_name}: T={steps}, delta={delta:g} -> error "
                f"({message})",
                flush=True,
            )
        else:
            successful[(steps, delta)] = value

    return successful, errors


# The experiment has two levels: one-step accountants depend only on (q,sigma),
# while final conversion/composition also depends on T and delta. The code
# reruns the SciPy-based calculation and FLINT-based calculation for every tuple of the parameters.
def run_experiment(args: argparse.Namespace) -> list[dict[str, Any]]:
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
        raise ValueError("order-window must be nonnegative")
    if args.birrell_taylor_order is not None and args.birrell_taylor_order < 3:
        raise ValueError("birrell-taylor-order must be at least 3")

    birrell_taylor_orders = (
        (3, 4, 5)
        if args.birrell_taylor_order is None
        else (args.birrell_taylor_order,)
    )

    # Wang et al. and Birrell et al. are evaluated at every integer alpha=2,...,alpha_max.
    orders = np.arange(2, args.alpha_max + 1, dtype=int)
    cases = [
        (steps, delta_text, delta)
        for steps in steps_values
        for delta_text, delta in zip(delta_texts, delta_values)
    ]
    # numerical_error is a bound on the final composed RDP inflation.
    numerical_error = _positive_fraction(args.numerical_error)

    rows: list[dict[str, Any]] = []
    for q_text, q in zip(q_texts, q_values):
        for sigma_text, sigma in zip(sigma_texts, sigma_values):
            shift = 2.0 / sigma
            print(
                f"\nq={q:g}, sigma={sigma:g}, Delta={shift:g}, "
                f"orders=2,...,{args.alpha_max}",
                flush=True,
            )

            print(
                "  Independent SciPy binary search and FLINT calculation "
                "for each (T, delta) case",
                flush=True,
            )
            # These dictionaries are recreated for every (q,sigma).
            scipy_results = {}
            windows = {}
            conversions = {}
            # First select alpha numerically by minimizing the improved final
            # RDP-to-DP objective. Then calculate (based on FLINT) the one-step RDP independently
            # on the requested window around that selected alpha.
            for steps, delta_text, delta in cases:
                search = find_optimal_order_binary(
                    q,
                    sigma,
                    steps,
                    delta,
                    accuracy=float(args.scipy_accuracy),
                    alpha_max=args.alpha_max,
                )
                scipy_results[(steps, delta_text)] = search
                boundary = " (alpha_max boundary)" if search.hit_upper_boundary else ""
                window = _order_window(
                    int(search.optimal_alpha),
                    args.order_window,
                    args.alpha_max,
                )
                windows[(steps, delta_text)] = window

                # RDP adds over T steps. Giving the one-step maximizer tolerance
                # numerical_error/T ensures its total possible inflation after
                # multiplication by T is at most numerical_error.
                # Each tuple receives the full requested total RDP numerical
                # error budget and is evaluated independently of every other T.
                one_step_tolerance = numerical_error / steps
                rdp = _calculate_orders(
                    q_text,
                    sigma_text,
                    window,
                    one_step_tolerance,
                )
                # Take the smallest epsilon based on calculations from different orders.
                conversions[(steps, delta_text)] = min(
                    (
                        rdp_to_dp_at_order(
                            alpha,
                            rdp[alpha],
                            steps,
                            delta_text,
                        )
                        for alpha in window
                    ),
                    key=lambda result: result.epsilon,
                )
                print(
                    f"    T={steps}, delta={delta_text}: alpha="
                    f"{search.optimal_alpha}{boundary}; queried orders "
                    f"{list(search.evaluated_orders)}; window {window}",
                    flush=True,
                )

            # Wang et al. and Birrell et al. return one-step RDP profiles on the full order grid.
            # Composition is done by multiplication by T, followed by the same improved
            # conversion and minimization.
            print("  WBK profile via dp-accounting on all orders", flush=True)
            wbk_rdp = wbk_profile(orders, sigma, q_text)
            wbk_results = {
                (steps, delta_text): rdp_to_dp(orders, wbk_rdp, steps, delta)
                for steps, delta_text, delta in cases
            }

            # By default, retain the smallest final epsilon among Taylor order m=3,4,5.
            print(
                "  Birrell et al. profiles on all orders for m="
                + ",".join(str(m) for m in birrell_taylor_orders),
                flush=True,
            )
            birrell_rdp_by_taylor_order = {
                m: birrell_profile(
                    orders,
                    noise_multiplier=sigma,
                    sampling_probability=q,
                    taylor_order=m,
                )
                for m in birrell_taylor_orders
            }

            lossless_cases = list(
                dict.fromkeys((steps, delta) for steps, _, delta in cases)
            )

            print(
                "  Connect-the-Dots composition via Google's PLD accountant",
                flush=True,
            )
            connect_the_dots_results, connect_the_dots_errors = (
                _run_with_case_fallback(
                    "Connect-the-Dots",
                    lossless_cases,
                    batch_calculation=lambda: connect_the_dots_epsilon_upper_bounds(
                        q,
                        sigma,
                        steps_values,
                        delta_values,
                    ),
                    single_calculation=lambda steps, delta: (
                        connect_the_dots_epsilon_upper_bounds(
                            q,
                            sigma,
                            [steps],
                            [delta],
                        )
                    ),
                    extract_case=lambda output, steps, delta: output[(steps, delta)],
                    validate=_validate_ctd_result,
                )
            )

            # Gather all guarantees for each full tuple.
            for steps, delta_text, target_delta in cases:
                scipy = scipy_results[(steps, delta_text)]
                alpha = int(scipy.optimal_alpha)
                detail = scipy.optimal_detail
                conversion = conversions[(steps, delta_text)]
                birrell_taylor_order, birrell = min(
                    (
                        (
                            m,
                            rdp_to_dp(
                                orders,
                                profile,
                                steps,
                                target_delta,
                            ),
                        )
                        for m, profile in birrell_rdp_by_taylor_order.items()
                    ),
                    key=lambda item: item[1].epsilon,
                )
                wbk = wbk_results[(steps, delta_text)]
                lossless_key = (steps, target_delta)
                if lossless_key not in connect_the_dots_results:
                    raise RuntimeError(
                        "Connect-the-Dots failed for "
                        f"q={q:g}, sigma={sigma:g}, T={steps}, "
                        f"delta={target_delta:g}. "
                        + connect_the_dots_errors.get(
                            lossless_key, "unknown error"
                        )
                    )

                connect_the_dots = connect_the_dots_results[lossless_key]

                row = {
                    "sampling_probability_q": q,
                    "noise_multiplier_sigma": sigma,
                    "replacement_shift_Delta": shift,
                    "steps": steps,
                    "nominal_epochs_Tq": steps * q,
                    "privacy_delta": target_delta,
                    "isosceles_epsilon_upper": conversion.epsilon,
                    "isosceles_epsilon_upper_text": (
                        conversion.upper_text
                    ),
                    "isosceles_scipy_epsilon_estimate": scipy.epsilon,
                    "isosceles_scipy_optimal_alpha": alpha,
                    "isosceles_scipy_best_u": float(detail.maximizing_u),
                    "isosceles_scipy_one_step_estimated_error": float(
                        detail.estimated_error
                    ),
                    # Retained for CSV compatibility. Under binary search,
                    # this means all SciPy orders queried for this case converged.
                    "isosceles_scipy_all_orders_converged": (
                        scipy.all_evaluated_orders_converged
                    ),
                    "birrell_epsilon": birrell.epsilon,
                    "birrell_optimal_alpha": birrell.optimal_alpha,
                    "birrell_taylor_order": birrell_taylor_order,
                    "wbk_epsilon": wbk.epsilon,
                    "wbk_optimal_alpha": wbk.optimal_alpha,
                    "alpha_max": args.alpha_max,
                    "total_rdp_numerical_error_budget": float(
                        numerical_error
                    ),
                    "scipy_requested_accuracy": float(args.scipy_accuracy),
                    "rdp_conversion": "improved standard RDP-to-DP conversion",
                    "connect_the_dots_epsilon_upper": connect_the_dots,
                }
                rows.append(row)

    rows.sort(
        key=lambda row: (
            row["sampling_probability_q"],
            row["noise_multiplier_sigma"],
            row["steps"],
            row["privacy_delta"],
        )
    )

    # Writing occurs after every tuple has at least one successful lossless accountant.
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "dpsgd_comparison.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

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
                    "connect_the_dots_epsilon_upper",
                )
            },
        }
        for row in rows
    ]
    _print_table(display_rows)
    print(f"\nWrote {output_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q",
        default="0.01",
        help="batch fraction b/n; comma-separated values are allowed",
    )
    parser.add_argument("--noise-multipliers", default="2,4,8")
    parser.add_argument("--steps", default="1000,5000,10000")
    parser.add_argument("--deltas", default="1e-6,1e-8")
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
            "number of integer orders on each side of the SciPy-selected "
            "order to calculate (default: 0)"
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
    parser.add_argument(
        "--birrell-taylor-order",
        type=int,
        default=None,
        help=(
            "Birrell Taylor order m; by default evaluate m=3,4,5 and retain "
            "the smallest final epsilon"
        ),
    )
    parser.add_argument("--output-directory", default="results")
    return parser


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
