"""Connect-the-dots PLD accounting for DP-SGD."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np
from scipy.special import ndtr

from pld import (
    fixed_size_replace_privacy_profile,
    privacy_loss_threshold,
)


# The exact one-step privacy profile is sampled at multiples of this gap.
# The support searches place both endpoints where the corresponding exact
# privacy-loss tails are small; Connect-the-dots then treats the endpoints
# pessimistically rather than deleting samples outside the finite grid.
_VALUE_DISCRETIZATION_INTERVAL = 1e-4
_TAIL_MASS_TRUNCATION = 1e-15
_MAX_GRID_POINTS = 2_000_001


def _smallest_index_with_tail_at_most(
    tail_probability: Callable[[int], float],
    target: float,
    max_index: int,
) -> int:
    """Find the smallest nonnegative grid index whose tail is at most target."""
    # First double the candidate index until the monotone tail probability is
    # below target; then binary search the bracket. This finds the shortest
    # one-sided support satisfying the requested exact tail bound without
    # scanning every possible grid point.
    if tail_probability(0) <= target:
        return 0

    lower = 0
    upper = 1
    while upper < max_index and tail_probability(upper) > target:
        lower = upper
        upper = min(2 * upper, max_index)

    if tail_probability(upper) > target:
        raise ValueError(
            "Connect-the-dots support exceeds max_grid_points; increase "
            "max_grid_points or use a coarser value_discretization_interval"
        )

    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if tail_probability(midpoint) <= target:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _rounded_support(
    sampling_probability: float,
    replacement_shift: float,
    discretization: float,
    tail_probability_bound: float,
    max_grid_points: int,
) -> tuple[int, int]:
    """Choose finite grid endpoints with small exact one-step PLD tails."""
    # A grid index i represents privacy loss i*discretization. Inverting the
    # monotone one-step likelihood ratio converts each endpoint to a Gaussian threshold.
    q = sampling_probability
    shift = replacement_shift

    def threshold(index: int) -> float:
        return float(privacy_loss_threshold(index * discretization, q, shift))

    # For the negative endpoint -a, the lower tail probability is
    # Pr[Y<-a]=Phi(-L^{-1}(a)).
    def lower_tail(index: int) -> float:
        return float(ndtr(-threshold(index)))

    # For the positive endpoint a, the upper tail probability is
    # Pr[Y>a]=(1-q)Phi(-L^{-1}(a))+q Phi(shift-L^{-1}(a)).
    def upper_tail(index: int) -> float:
        value = threshold(index)
        return float((1.0 - q) * ndtr(-value) + q * ndtr(shift - value))

    # Search the negative and positive sides independently.
    max_index = max_grid_points - 1
    lower_index = _smallest_index_with_tail_at_most(
        lower_tail,
        tail_probability_bound,
        max_index,
    )
    upper_index = _smallest_index_with_tail_at_most(
        upper_tail,
        tail_probability_bound,
        max_index,
    )
    grid_points = lower_index + upper_index + 1
    if grid_points > max_grid_points:
        raise ValueError(
            f"Connect-the-dots grid needs {grid_points:,} points, exceeding "
            f"max_grid_points={max_grid_points:,}"
        )
    return -lower_index, upper_index


def connect_the_dots_epsilon_upper_bounds(
    sampling_probability: float,
    noise_multiplier: float,
    steps: Iterable[int],
    deltas: Iterable[float],
    *,
    value_discretization_interval: float = _VALUE_DISCRETIZATION_INTERVAL,
    tail_mass_truncation: float = _TAIL_MASS_TRUNCATION,
    max_grid_points: int = _MAX_GRID_POINTS,
) -> dict[tuple[int, float], float]:
    """Return ``pessimistic`` epsilon bounds.

    The one-step hockey-stick divergence is evaluated on a uniform grid,
    converted with connect-the-dots construction, and then
    self-composed with the PLD implementation.
    """
    # All requested step counts use the same exact one-step profile, but each
    # final (steps, delta) query is read from the appropriately composed PLD.
    q = float(sampling_probability)
    sigma = float(noise_multiplier)
    step_values = sorted({int(value) for value in steps})
    delta_values = [float(value) for value in deltas]
    discretization = float(value_discretization_interval)
    tail_mass = float(tail_mass_truncation)
    max_points = int(max_grid_points)

    if not 0.0 < q < 1.0:
        raise ValueError("sampling_probability must lie in (0,1)")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("noise_multiplier must be positive and finite")
    if not step_values or step_values[0] <= 0:
        raise ValueError("steps must contain positive integers")
    if not delta_values or not all(0.0 < value < 1.0 for value in delta_values):
        raise ValueError("deltas must lie in (0,1)")
    if not math.isfinite(discretization) or discretization <= 0.0:
        raise ValueError("value_discretization_interval must be positive and finite")
    if not math.isfinite(tail_mass) or not 0.0 < tail_mass < 1.0:
        raise ValueError("tail_mass_truncation must lie in (0,1)")
    if max_points < 3:
        raise ValueError("max_grid_points must be at least 3")

    try:
        from dp_accounting.pld import pld_pmf, privacy_loss_distribution
    except ImportError as exc:  
        raise ImportError(
            "Install `dp-accounting==0.6.0` to use Connect-the-dots accounting."
        ) from exc

    # At the largest requested T, choose each one-step privacy-loss tail to be
    # at most min(delta)/(4T). In particular, the connect-the-dots infinity mass
    # is delta(epsilon_max), which is at most the positive privacy-loss
    # tail, so after T compositions it contributes at most T times this bound.
    # Google's self-composition also pessimistically adds its supplied truncation
    # allowance to infinity mass; limiting that allowance to min(delta)/4 keeps
    # both effects strictly below the smallest requested final delta.
    minimum_delta = min(delta_values)
    maximum_steps = max(step_values)
    one_step_tail_bound = min(
        tail_mass,
        minimum_delta / (4.0 * maximum_steps),
    )
    composition_tail_mass = min(tail_mass, minimum_delta / 4.0)
    if one_step_tail_bound <= 0.0 or composition_tail_mass <= 0.0:
        raise ValueError("requested delta is too small for floating-point accounting")

    # shift=2/sigma. The support routine returns
    # integer endpoints so every profile evaluation lies exactly on the fixed
    # privacy-loss spacing required by connect-the-dots.
    replacement_shift = 2.0 / sigma
    rounded_lower, rounded_upper = _rounded_support(
        q,
        replacement_shift,
        discretization,
        one_step_tail_bound,
        max_points,
    )
    # Evaluate the one-step hockey-stick divergence at each
    # grid location. connect-the-dots uses these values, rather than sampled
    # values of a privacy-loss density, to construct a pessimistic discrete PLD.
    rounded_epsilons = np.arange(
        rounded_lower,
        rounded_upper + 1,
        dtype=np.int64,
    )
    epsilon_values = rounded_epsilons.astype(float) * discretization
    profile_values = np.asarray(
        fixed_size_replace_privacy_profile(
            epsilon_values,
            q,
            replacement_shift,
        ),
        dtype=float,
    )

    # Consecutive differences of the exact profile values determine the finite
    # PMF masses in the connect-the-dots formula. Its hockey-stick
    # curve connects the supplied values pessimistically between grid points,
    # and the last value delta(epsilon_max) is assigned to privacy loss +infinity.
    one_step_pmf = pld_pmf.create_pmf_pessimistic_connect_dots_fixed_gap(
        discretization=discretization,
        rounded_epsilon_lower=rounded_lower,
        rounded_epsilon_upper=rounded_upper,
        deltas=profile_values,
    )
    one_step_pld = privacy_loss_distribution.PrivacyLossDistribution(one_step_pmf)

    # Self-composition adds independent one-step privacy losses by a power
    # convolution. Google chooses a finite output range whose total excluded
    # mass is bounded by composition_tail_mass, and pessimistically adds that
    # entire allowance to the +infinity atom. The final call then inverts the
    # composed hockey-stick curve at each target delta.
    output: dict[tuple[int, float], float] = {}
    for step_count in step_values:
        composed = one_step_pld.self_compose(
            num_times=step_count,
            tail_mass_truncation=composition_tail_mass,
        )
        for target_delta in delta_values:
            output[(step_count, target_delta)] = float(
                composed.get_epsilon_for_delta(target_delta)
            )
    return output
