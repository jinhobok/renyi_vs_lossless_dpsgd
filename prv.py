"""PRV accountant for fixed-size replace-one DP-SGD.

The package's built-in subsampled Gaussian PRV is for Poisson subsampling. This file
instead provides the artificial fixed-size replace-one privacy random variable
implemented in :mod:`pld`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from pld import (
    artificial_prv_rdp_upper,
    fixed_size_replace_prv_cdf,
)

try:
    from prv_accountant.privacy_random_variables import PrivacyRandomVariable
except ImportError:  
    PrivacyRandomVariable = object  


# PRVAccountant needs a CDF and an RDP upper bound for a single privacy random
# variable. This class provides those quantities for the subsampled Gaussian mechanism.
class FixedSizeReplaceOneGaussianPRV(PrivacyRandomVariable):
    """Custom privacy random variable consumed by ``PRVAccountant``."""

    def __init__(self, sampling_probability: float, noise_multiplier: float):
        if PrivacyRandomVariable is object:
            raise ImportError("Install `prv-accountant` to use the PRV comparison.")
        self.q = float(sampling_probability)
        self.sigma = float(noise_multiplier)
        if not 0.0 < self.q < 1.0:
            raise ValueError("sampling_probability must lie in (0,1)")
        if self.sigma <= 0.0:
            raise ValueError("noise_multiplier must be positive")
        self.shift = 2.0 / self.sigma

    # The accountant discretizes this CDF to approximate repeated sums of the
    # one-step privacy random variable.
    def cdf(self, value: Any) -> Any:
        return fixed_size_replace_prv_cdf(value, self.q, self.shift)

    # The package queries RDP while choosing a sufficiently wide numerical
    # domain. artificial_prv_rdp_upper rounds a noninteger request upward to
    # an integer order (apart from its documented 1e-12 snapping tolerance)
    # and evaluates the corresponding finite-sum upper bound. Caching is local
    # to this PRV instance and avoids recomputing an identical request.
    @lru_cache(maxsize=256)
    def rdp(self, alpha: float) -> float:
        return artificial_prv_rdp_upper(alpha, self.q, self.shift)


def prv_epsilon_bounds(
    sampling_probability: float,
    noise_multiplier: float,
    steps: Iterable[int],
    deltas: Iterable[float],
    *,
    epsilon_error: float = 0.02,
    delta_error: float | None = None,
) -> dict[tuple[int, float], tuple[float, float, float]]:
    """Return PRV ``(lower, estimate, upper)`` epsilon values."""
    try:
        from prv_accountant import PRVAccountant
    except ImportError as exc:
        raise ImportError("Install `prv-accountant`.") from exc

    step_values = [int(value) for value in steps]
    delta_values = [float(value) for value in deltas]
    if not step_values or min(step_values) <= 0:
        raise ValueError("steps must contain positive integers")
    if not delta_values or not all(0.0 < value < 1.0 for value in delta_values):
        raise ValueError("deltas must lie in (0,1)")

    delta_error = (
        min(delta_values) / 1000.0 if delta_error is None else float(delta_error)
    )
    accountant = PRVAccountant(
        prvs=[FixedSizeReplaceOneGaussianPRV(sampling_probability, noise_multiplier)],
        max_self_compositions=[max(step_values)],
        eps_error=float(epsilon_error),
        delta_error=delta_error,
    )

    # Each query returns the accountant's lower estimate, central estimate, and
    # upper estimate for the final epsilon at the specified (steps, delta).
    output = {}
    for step_count in step_values:
        for delta in delta_values:
            bounds = accountant.compute_epsilon(
                delta=delta,
                num_self_compositions=[step_count],
            )
            output[(step_count, delta)] = tuple(float(value) for value in bounds)
    return output