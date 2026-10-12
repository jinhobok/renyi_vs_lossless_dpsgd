"""Convert Rényi differential privacy guarantees to ``(epsilon, delta)``-DP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


# epsilon is the converted final guarantee, optimal_alpha records the order
# attaining the smallest tested value, and upper_text is populated only when
# the conversion was carried out with Arb arithmetic.
@dataclass(frozen=True)
class EpsilonResult:
    epsilon: float
    optimal_alpha: int
    upper_text: str | None = None


def rdp_to_dp_epsilon_at_order(
    alpha: int,
    one_step_rdp: float,
    steps: int,
    target_delta: float,
    *,
    clip_at_zero: bool = True,
) -> float:
    """Convert one floating-point RDP order to epsilon."""
    order = int(alpha)
    if order < 2:
        raise ValueError("alpha must be an integer at least 2")
    if steps <= 0:
        raise ValueError("steps must be positive")
    delta = float(target_delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("target_delta must lie in (0,1)")
    rdp = float(one_step_rdp)
    if not math.isfinite(rdp):
        raise ValueError("one_step_rdp must be finite")

    # RDP composes additively, so T identical steps have RDP T*rho(alpha).
    # The improved conversion used throughout this project is
    #
    #   epsilon_alpha(delta)
    #     = T*rho(alpha) + log(1-1/alpha)
    #       - log(delta*alpha)/(alpha-1).
    #
    # The raw, unclipped value is useful for comparing neighboring orders.
    order_float = float(order)
    epsilon = (
        steps * rdp
        + math.log1p(-1.0 / order_float)
        - (math.log(delta) + math.log(order_float)) / (order_float - 1.0)
    )
    return max(0.0, epsilon) if clip_at_zero else epsilon


def rdp_to_dp(
    orders: Sequence[int] | np.ndarray,
    one_step_rdp: Sequence[float] | np.ndarray,
    steps: int,
    target_delta: float,
) -> EpsilonResult:
    """Compose a floating RDP profile and minimize over its orders."""
    # Vectorize the same fixed-order formula over a complete one-step RDP
    # profile. Each array position represents one candidate Renyi order.
    orders = np.asarray(orders, dtype=float)
    rdp = np.asarray(one_step_rdp, dtype=float)
    if orders.shape != rdp.shape:
        raise ValueError("orders and one_step_rdp must have the same shape")
    if np.any(orders <= 1.0):
        raise ValueError("all orders must exceed one")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < target_delta < 1.0:
        raise ValueError("target_delta must lie in (0,1)")

    # Add T copies of the one-step RDP value and then apply the improved
    # RDP-to-(epsilon,delta) conversion at every candidate order.
    epsilon = (
        steps * rdp
        + np.log1p(-1.0 / orders)
        - (math.log(target_delta) + np.log(orders)) / (orders - 1.0)
    )
    # Minimizing over alpha is valid because every order yields a separate
    # DP guarantee; retaining the smallest epsilon gives the tightest one in
    # the supplied order grid.
    best = int(np.nanargmin(epsilon))
    return EpsilonResult(
        epsilon=max(0.0, float(epsilon[best])),
        optimal_alpha=int(orders[best]),
    )


def rdp_to_dp_at_order(
    alpha: int,
    one_step_rdp_upper: Any,
    steps: int,
    target_delta: str,
    *,
    dps: int = 100,
) -> EpsilonResult:
    """Compose and convert one certified RDP order using Arb arithmetic."""
    try:
        from flint import arb, ctx
        from utils import restore_precision, upper_text, upward_float
    except ImportError as exc: 
        raise ImportError("certified conversion requires python-flint") from exc

    if alpha < 2:
        raise ValueError("alpha must be an integer at least 2")
    if steps <= 0:
        raise ValueError("steps must be positive")

    # All following operations use Arb balls. The user-supplied decimal delta
    # is parsed directly, and the precision is restored on exit.
    with restore_precision():
        ctx.dps = max(60, int(dps))
        delta = arb(str(target_delta))
        if delta.lower() <= 0 or delta.upper() >= 1:
            raise ValueError("target_delta must lie in (0,1)")

        # Use the upper endpoint of the certified one-step RDP interval before
        # multiplying by T. Monotonicity of the conversion in rho(alpha) then
        # makes the resulting epsilon an upper bound. Taking .upper() after
        # the full expression also rounds every operation in
        # the pessimistic direction.
        order = arb(alpha)
        epsilon = (
            one_step_rdp_upper.upper() * steps
            + (1 - 1 / order).log()
            - (delta * order).log() / (order - 1)
        ).upper()
        # If the analytical expression is negative, replacing it by zero only
        # enlarges epsilon and therefore weakens that particular guarantee.
        if epsilon < 0:
            epsilon = arb(0)

        # Convert the exact Arb upper endpoint to an outward-rounded decimal,
        # then to a binary float without ever rounding below that decimal.
        text = upper_text(epsilon, digits=70)
        return EpsilonResult(
            epsilon=upward_float(text),
            optimal_alpha=alpha,
            upper_text=text,
        )
