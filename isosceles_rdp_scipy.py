"""SciPy estimate of the RDP maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp as _scipy_logsumexp, roots_hermitenorm

from rdp_to_dp import rdp_to_dp_epsilon_at_order


# Grid points are evaluated together to avoid thousands of small Python/SciPy calls.
_GRID_BATCH_SIZE = 32
_LOW_ACCURACY_THRESHOLD = 1e-6


@dataclass(frozen=True)
class EstimatedMaximum:
    value: float
    maximizing_u: float
    estimated_error: float
    quadrature_nodes: int
    grid_intervals: int
    converged: bool
    used_u_zero_shortcut: bool = False


def integer_u_zero_renyi(
    sampling_probability: float,
    replacement_shift: float,
    alpha: int,
) -> float:
    """Evaluate the exact integer-order RDP value at ``u=0`` in floats.
    """
    p = float(sampling_probability)
    shift = float(replacement_shift)
    order = int(alpha)
    if not 0.0 < p < 1.0:
        raise ValueError("sampling_probability must lie in (0,1)")
    if shift <= 0.0:
        raise ValueError("replacement_shift must be positive")
    if order < 2:
        raise ValueError("alpha must be an integer at least 2")

    # For integer alpha, the Renyi moment is
    #
    #   sum_{k=0}^alpha binom(alpha,k)(1-p)^(alpha-k)p^k
    #       * exp(k(k-1)Delta^2/2).
    #
    # Taking logsumexp and dividing by alpha-1 gives D_alpha.
    k = np.arange(order + 1, dtype=float)
    log_terms = (
        gammaln(order + 1.0)
        - gammaln(k + 1.0)
        - gammaln(order - k + 1.0)
        + (order - k) * math.log1p(-p)
        + k * math.log(p)
        + 0.5 * k * (k - 1.0) * shift * shift
    )
    return max(0.0, float(_scipy_logsumexp(log_terms)) / (order - 1.0))


def u_zero_endpoint_condition(
    sampling_probability: float,
    replacement_shift: float,
    alpha: int,
) -> bool:
    """A sufficient condition for a unique maximum at ``u=0``.

    Write ``p`` for ``sampling_probability`` and ``Delta`` for
    ``replacement_shift``. With

        x = p (alpha - 1) Delta^2 / 2,

    the condition tested here is

        p exp((alpha-1)(1-p)Delta^2/2)
        >= alpha/(alpha-1).

    This routine is used only for the scipy-based calculations.
    """
    p = float(sampling_probability)
    shift = float(replacement_shift)
    order = int(alpha)
    if not 0.0 < p < 1.0:
        raise ValueError("sampling_probability must lie in (0,1)")
    if shift <= 0.0:
        raise ValueError("replacement_shift must be positive")
    if order < 2:
        raise ValueError("alpha must be an integer at least 2")

    # Compare the logarithms of the two positive sides in the inequality above. 
    order_gap = float(order - 1)
    shift_squared = shift * shift

    log_left = math.log(p) + 0.5 * order_gap * (1.0 - p) * shift_squared
    log_right = math.log(float(order) / order_gap)

    return log_left >= math.nextafter(log_right, math.inf)


# roots_hermitenorm integrates against exp(-x^2/2). Dividing its weights by
# sqrt(2*pi) turns the rule into an approximation of expectation under a
# standard normal variable, which is the remaining one-dimensional integral.
def _gauss_hermite_rule(quadrature_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return standard-normal Gauss--Hermite nodes and log weights."""
    nodes, weights = roots_hermitenorm(int(quadrature_nodes))
    keep = weights > 0.0
    kept_nodes = np.asarray(nodes[keep], dtype=float)
    log_weights = (
        np.log(np.asarray(weights[keep], dtype=float))
        - 0.5 * math.log(2.0 * math.pi)
    )
    kept_nodes.setflags(write=False)
    log_weights.setflags(write=False)
    return kept_nodes, log_weights


def _logsumexp_inplace(values: np.ndarray, axis: int) -> np.ndarray:
    """NumPy log-sum-exp specialized for finite float arrays."""
    maximum = np.max(values, axis=axis, keepdims=True)
    if not np.all(np.isfinite(maximum)):
        return np.asarray(_scipy_logsumexp(values, axis=axis))

    values -= maximum
    np.exp(values, out=values)
    result = np.log(np.sum(values, axis=axis))
    result += np.squeeze(maximum, axis=axis)
    return np.asarray(result)


class IntegerIsoscelesRDP:
    """Gauss--Hermite evaluator for one integer order."""

    def __init__(
        self,
        sampling_probability: float,
        replacement_shift: float,
        alpha: int,
        quadrature_nodes: int,
    ) -> None:
        self.q = float(sampling_probability)
        self.shift = float(replacement_shift)
        self.alpha = int(alpha)
        self.quadrature_nodes = int(quadrature_nodes)
        if not 0.0 < self.q < 1.0:
            raise ValueError("sampling_probability must lie in (0,1)")
        if self.shift <= 0.0:
            raise ValueError("replacement_shift must be positive")
        if self.alpha < 2:
            raise ValueError("alpha must be an integer at least 2")
        if self.quadrature_nodes < 16:
            raise ValueError("quadrature_nodes must be at least 16")

        # Expand the alpha-th power of the numerator mixture over k selected
        # shifted components. Terms independent of u and the quadrature node
        # are precomputed once for this alpha.
        self.nodes, self.log_weights = _gauss_hermite_rule(self.quadrature_nodes)
        self.k = np.arange(self.alpha + 1, dtype=float)
        self.log_binomial = (
            gammaln(self.alpha + 1.0)
            - gammaln(self.k + 1.0)
            - gammaln(self.alpha - self.k + 1.0)
        )
        # For the isosceles parameterization, the second shift has length
        # beta=Delta*sqrt(2u). Conditioning on its standard-normal coordinate
        # leaves the k-th numerator term with
        #
        #   exp(k*beta*V/2 - k^2*beta^2/8)
        #
        # in addition to the coefficient above.
        self._quadratic = 0.125 * self.k * self.k

        self._log_q0 = math.log1p(-self.q)
        self._log_q = math.log(self.q)
        self._shift_squared = self.shift * self.shift
        self._inverse_order_gap = 1.0 / (self.alpha - 1.0)
        self._base_coefficients = (
            self.log_binomial
            + (self.alpha - self.k) * self._log_q0
            + self.k * self._log_q
            + 0.5 * self.k * (self.k - 1.0) * self._shift_squared
        )
        self._linear_node_coefficients = (
            0.5 * self.k[:, None] * self.nodes[None, :]
        )

    # This is the same u=0 finite sum as integer_u_zero_renyi, reusing the
    # coefficients already stored by the evaluator.
    def value_at_u_zero(self) -> float:
        """Evaluate the exact finite integer-order sum at ``u=0`` in floats."""
        log_moment = float(_scipy_logsumexp(self._base_coefficients))
        return max(0.0, log_moment * self._inverse_order_gap)

    def _values_at_u(self, u_values: np.ndarray) -> np.ndarray:
        """Vectorized internal evaluator for a one-dimensional array of u's."""
        values = np.asarray(u_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("u_values must be one-dimensional")
        if np.any((values < 0.0) | (values > 0.5)):
            raise ValueError("u must lie in [0,1/2]")

        # Evaluate u values in batches. 
        # The one-dimensional integral (expectation) is given as
        #
        #   E_V[
        #     sum_{k=0}^alpha binom(alpha,k) (1-q)^(alpha-k) q^k
        #       * exp(k*beta*V/2
        #             + k*(k-1)*Delta^2/2
        #             - k^2*beta^2/8)
        #     / ((1-q)+q*exp(beta*V-beta^2/2))^(alpha-1)
        #   ],
        #
        # with V ~ N(0,1).
        output = np.empty(values.size, dtype=float)
        for start in range(0, values.size, _GRID_BATCH_SIZE):
            stop = min(values.size, start + _GRID_BATCH_SIZE)
            u = values[start:stop]
            beta = self.shift * np.sqrt(2.0 * u)
            beta_squared = beta * beta

            # First sum the alpha+1 numerator terms for every quadrature node.
            log_terms = (
                beta[:, None, None]
                * self._linear_node_coefficients[None, :, :]
            )
            log_terms += self._base_coefficients[None, :, None]
            log_terms -= (
                beta_squared[:, None, None]
                * self._quadratic[None, :, None]
            )
            log_numerator = _logsumexp_inplace(log_terms, axis=1)

            # log_b is the logarithm of the denominator mixture density divided
            # by the reference N(0,I) density at the conditioned coordinate.
            log_b = np.logaddexp(
                self._log_q0,
                self._log_q
                + beta[:, None] * self.nodes[None, :]
                - 0.5 * beta_squared[:, None],
            )
            # Multiply by B^(1-alpha), add the N(0,1) quadrature
            # weights, sum over nodes, then take log/(alpha-1).
            log_integrand = log_numerator - (self.alpha - 1.0) * log_b
            log_integrand += self.log_weights[None, :]
            log_moment = _logsumexp_inplace(log_integrand, axis=1)
            output[start:stop] = np.maximum(
                0.0,
                log_moment * self._inverse_order_gap,
            )
        return output

    def value_at_u(self, u: float) -> float:
        """Estimate the Renyi divergence at ``u`` in ``[0, 1/2]``."""
        u = float(u)
        if not 0.0 <= u <= 0.5:
            raise ValueError("u must lie in [0,1/2]")
        if u == 0.0:
            return self.value_at_u_zero()

        # The grid search above evaluates many u values simultaneously.
        # SciPy's local optimizer instead calls the objective at one u at a time,
        # so this function evaluates the same RDP formula for a single u.
        beta = self.shift * math.sqrt(2.0 * u)
        beta_squared = beta * beta
        log_terms = beta * self._linear_node_coefficients
        log_terms += self._base_coefficients[:, None]
        log_terms -= beta_squared * self._quadratic[:, None]
        log_numerator = _logsumexp_inplace(log_terms, axis=0)

        log_b = np.logaddexp(
            self._log_q0,
            self._log_q + beta * self.nodes - 0.5 * beta_squared,
        )
        log_integrand = (
            self.log_weights
            + log_numerator
            - (self.alpha - 1.0) * log_b
        )
        log_moment = float(_logsumexp_inplace(log_integrand, axis=0))
        return max(0.0, log_moment * self._inverse_order_gap)

    def maximum_on_grid(self, grid_intervals: int) -> tuple[float, float]:
        """Search a grid, then optimize around every grid-local maximum."""
        return self._maximum_on_grid(grid_intervals, 1e-12)

    def _maximum_on_grid(
        self,
        grid_intervals: int,
        xatol: float,
    ) -> tuple[float, float]:
        intervals = int(grid_intervals)
        if intervals < 4:
            raise ValueError("grid_intervals must be at least 4")

        # A uniform grid records the endpoints and every grid point that
        # is a local maximum among its sampled neighbors. Each interior candidate
        # is then refined only inside those two neighboring grid cells.
        grid = np.linspace(0.0, 0.5, intervals + 1)
        values = self._values_at_u(grid)

        # Suppress sampled plateaus whose differences are only at roundoff scale;
        # the global grid argmax is retained separately in every case.
        scale = max(1.0, float(np.max(np.abs(values))))
        tolerance = 64.0 * np.finfo(float).eps * scale
        middle = values[1:-1]
        left = values[:-2]
        right = values[2:]
        local_maxima = np.flatnonzero(
            (middle >= left - tolerance)
            & (middle >= right - tolerance)
            & ((middle > left + tolerance) | (middle > right + tolerance))
        ) + 1
        candidates = {int(np.argmax(values)), *map(int, local_maxima)}

        # Retain the best grid endpoint as a candidate; bounded minimization of
        # -(Renyi divergence) refines each interior candidate without leaving its bracket.
        best_index = int(np.argmax(values))
        best_value = float(values[best_index])
        best_u = float(grid[best_index])
        for index in sorted(candidates):
            if index in (0, intervals):
                continue
            result = minimize_scalar(
                lambda value: -self.value_at_u(float(value)),
                bounds=(float(grid[index - 1]), float(grid[index + 1])),
                method="bounded",
                options={"xatol": float(xatol)},
            )
            value = float(-result.fun)
            if value > best_value:
                best_value, best_u = value, float(result.x)
        return best_value, best_u


def estimate_max_renyi(
    sampling_probability: float,
    replacement_shift: float,
    alpha: int,
    *,
    accuracy: float = 1e-8,
    initial_nodes: int = 48,
    initial_grid: int = 24,
    max_nodes: int = 384,
    max_grid: int = 768,
) -> EstimatedMaximum:
    """Refine quadrature and grid sizes until successive rounds agree."""
    if accuracy <= 0.0:
        raise ValueError("accuracy must be positive")

    # When the sufficient condition for u being argmax holds, the quadrature calculation and branch-and-bound optimization are skipped.
    if u_zero_endpoint_condition(
        sampling_probability,
        replacement_shift,
        alpha,
    ):
        return EstimatedMaximum(
            value=integer_u_zero_renyi(
                sampling_probability,
                replacement_shift,
                alpha,
            ),
            maximizing_u=0.0,
            estimated_error=0.0,
            quadrature_nodes=0,
            grid_intervals=0,
            converged=True,
            used_u_zero_shortcut=True,
        )

    # Otherwise double both the normal quadrature size and the u-grid size.
    nodes = int(initial_nodes)
    grid = int(initial_grid)
    previous_value: float | None = None
    stable_rounds = 0
    required_stable_rounds = 1 if accuracy >= _LOW_ACCURACY_THRESHOLD else 2
    xatol = (
        max(1e-12, min(1e-6, 0.1 * accuracy))
        if required_stable_rounds == 1
        else 1e-12
    )
    estimated_error = math.inf

    while True:
        evaluator = IntegerIsoscelesRDP(
            sampling_probability,
            replacement_shift,
            alpha,
            nodes,
        )
        value, maximizing_u = evaluator._maximum_on_grid(grid, xatol)

        # The absolute change in the estimated maximum combines quadrature,
        # grid, and local optimization changes.
        if previous_value is not None:
            estimated_error = abs(value - previous_value)
            stable_rounds = stable_rounds + 1 if estimated_error <= accuracy else 0
            if stable_rounds >= required_stable_rounds:
                return EstimatedMaximum(
                    value,
                    maximizing_u,
                    estimated_error,
                    nodes,
                    grid,
                    True,
                )
        previous_value = value

        if nodes >= max_nodes and grid >= max_grid:
            return EstimatedMaximum(
                value,
                maximizing_u,
                estimated_error,
                nodes,
                grid,
                False,
            )
        nodes = min(max_nodes, 2 * nodes)
        grid = min(max_grid, 2 * grid)


@dataclass(frozen=True)
class EstimatedOrderSearch:
    """Result of one independent SciPy binary search over integer orders."""

    epsilon: float
    optimal_alpha: int
    optimal_detail: EstimatedMaximum
    evaluated_orders: tuple[int, ...]
    all_evaluated_orders_converged: bool
    hit_upper_boundary: bool


def find_optimal_order_binary(
    sampling_probability: float,
    noise_multiplier: float,
    steps: int,
    target_delta: float,
    *,
    accuracy: float = 1e-6,
    alpha_min: int = 2,
    alpha_max: int = 64,
    verify_neighbors: bool = True,
) -> EstimatedOrderSearch:
    """Select the best integer order by an independent binary search.

    The RDP-to-DP objective is unimodal over the integer orders. Thus comparing the objective at ``m``
    and ``m+1`` determines which half of the current interval can contain its
    leftmost minimizer.
    """
    q = float(sampling_probability)
    sigma = float(noise_multiplier)
    requested_accuracy = float(accuracy)
    lower = int(alpha_min)
    upper = int(alpha_max)
    if not 0.0 < q < 1.0:
        raise ValueError("sampling_probability must lie in (0,1)")
    if sigma <= 0.0:
        raise ValueError("noise_multiplier must be positive")
    if requested_accuracy <= 0.0:
        raise ValueError("accuracy must be positive")
    if lower < 2:
        raise ValueError("alpha_min must be at least 2")
    if upper < lower:
        raise ValueError("alpha_max must be at least alpha_min")
    if steps <= 0:
        raise ValueError("steps must be positive")
    delta = float(target_delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("target_delta must lie in (0,1)")

    replacement_shift = 2.0 / sigma
    details: dict[int, EstimatedMaximum] = {}
    queried: set[int] = set()

    def estimate(alpha: int) -> EstimatedMaximum:
        order = int(alpha)
        if order < 2:
            raise ValueError("alpha must be an integer at least 2")
        if order not in details:
            details[order] = estimate_max_renyi(
                q,
                replacement_shift,
                order,
                accuracy=requested_accuracy,
            )
        return details[order]

    # For each queried alpha, first maximize the one-step RDP over u and then
    # evaluate the final raw objective
    #
    #   T*rho(alpha)+log(1-1/alpha)-log(delta*alpha)/(alpha-1).
    def objective(alpha: int) -> float:
        queried.add(alpha)
        detail = estimate(alpha)
        return rdp_to_dp_epsilon_at_order(
            alpha,
            detail.value,
            steps,
            delta,
            clip_at_zero=False,
        )

    # For the exact profile this integer objective is unimodal.
    # Here the comparisons use numerical estimates.
    lo, hi = lower, upper
    while lo < hi:
        middle = (lo + hi) // 2
        if objective(middle) <= objective(middle + 1):
            hi = middle
        else:
            lo = middle + 1

    # Recheck the selected order and immediate neighbors.
    selected = lo
    if verify_neighbors:
        candidates = range(
            max(lower, selected - 1),
            min(upper, selected + 1) + 1,
        )
        selected = min(candidates, key=lambda alpha: (objective(alpha), alpha))

    # Choose the Renyi order using the unmodified converted epsilon values.
    # Some converted values can be negative. If we replaced every negative value
    # by 0 before taking the minimum, all negative values would become tied and we
    # could lose the order that actually gives the smallest converted value.
    # After the minimizing order is chosen, replace its reported epsilon by
    # max(0, epsilon).
    raw_epsilon = objective(selected)
    detail = estimate(selected)
    evaluated_orders = tuple(sorted(queried))
    return EstimatedOrderSearch(
        epsilon=max(0.0, raw_epsilon),
        optimal_alpha=selected,
        optimal_detail=detail,
        evaluated_orders=evaluated_orders,
        all_evaluated_orders_converged=all(
            details[alpha].converged for alpha in evaluated_orders
        ),
        hit_upper_boundary=(selected == upper),
    )