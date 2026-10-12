"""Maximum of the Gaussian-mixture Rényi divergence over u.

For an integer order ``alpha >= 2`` this module calculates

    max_{0 <= u <= 1/2} D_alpha(P_a || P_b(u)).

It calls ``gaussian_mixture_renyi`` for point values,
controls every interval between sampled points by a bound on ``|D_alpha''(u)|``, 
and adaptively bisects only the interval with the
largest possible value. The returned upper endpoint is within the requested
additive tolerance of the true maximum.

Only integer Rényi orders are implemented in this file.
Dependency: ``pip install python-flint``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from math import comb
from typing import Any

from flint import arb, ctx

from utils import (
    Scalar,
    arb_input as _arb_input,
    exact_arb_fraction as _exact_arb_fraction,
    fraction_decimal_text,
    fraction_text as _fraction_text,
    integer_order as _integer_order,
    lower_text as _lower_text,
    restore_precision,
    upper_text as _upper_text,
    upper_text_with_tolerance,
)

try:
    from gaussian_mixture_renyi import (
        RenyiInterval,
        renyi,
    )
except ImportError as exc:
    raise ImportError(
        "gaussian_mixture_max_renyi.py requires "
        "gaussian_mixture_renyi.py"
    ) from exc


def _max_upper(x: arb, y: arb) -> arb:
    x, y = x.upper(), y.upper()
    return x if x >= y else y


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


# lower is the best sampled lower value and upper is the largest
# remaining interval envelope.
@dataclass(frozen=True)
class RenyiMaximumCurvature:
    lower: arb
    upper: arb
    additive_tolerance: arb
    p: arb
    delta: arb
    alpha: arb
    best_sample_u: Fraction
    point_evaluations: int
    processed_boxes: int
    unresolved_boxes_at_return: int
    maximum_depth_reached: int
    curvature_upper: arb
    used_u_zero_shortcut: bool = False

    @property
    def width(self) -> arb:
        return (self.upper - self.lower).upper()

    def upper_string(self, digits: int = 30) -> str:
        return upper_text_with_tolerance(
            self.upper, self.lower, self.additive_tolerance, digits
        )

    def lower_string(self, digits: int = 30) -> str:
        return _lower_text(self.lower, digits)

    def width_string(self, digits: int = 30) -> str:
        """Interval width in ordinary fixed-point notation."""
        return _upper_text(self.width, digits)

    def curvature_upper_string(self, digits: int = 30) -> str:
        """Curvature bound in ordinary fixed-point notation."""
        return _upper_text(self.curvature_upper, digits)

    def best_sample_u_string(self, digits: int = 30) -> str:
        return fraction_decimal_text(self.best_sample_u, digits)


@dataclass(frozen=True)
class _Box:
    left: Fraction
    right: Fraction
    upper: arb
    depth: int


# Degenerate mechanisms have a maximum independent of u.
def _closed_form_result(
    value: arb,
    p: arb,
    delta: arb,
    alpha: arb,
    tolerance: arb,
) -> RenyiMaximumCurvature:
    zero = arb(0)
    return RenyiMaximumCurvature(
        lower=value.lower(),
        upper=value.upper(),
        additive_tolerance=tolerance.lower(),
        p=p,
        delta=delta,
        alpha=alpha,
        best_sample_u=Fraction(0),
        point_evaluations=0,
        processed_boxes=0,
        unresolved_boxes_at_return=0,
        maximum_depth_reached=0,
        curvature_upper=zero,
    )


def _u_zero_endpoint_condition_holds(
    p: arb,
    delta: arb,
    integer_order: int,
) -> bool:
    """Uses a sufficient condition for the maximum to occur at u=0:

        p exp((alpha-1)(1-p)Delta^2/2)
        >= alpha/(alpha-1).
    """
    # Every operation is directed outward: the left side is rounded down and
    # the right side up.
    order_gap = arb(integer_order - 1)
    alpha = arb(integer_order)
    delta_squared = delta * delta
    x = (p * order_gap * delta_squared / 2).lower()
    if x.lower() <= 0:
        return False

    left = (
        p * (order_gap * (1 - p) * delta_squared / 2).exp()
    ).lower()
    right = (alpha / order_gap).upper()
    return left >= right


# Once the sufficient condition proves u=0 is the unique global maximizer, the
# point calculation u=0 is already a global maximum.
def _u_zero_result(
    point: RenyiInterval,
    p: arb,
    delta: arb,
    alpha: arb,
    tolerance: arb,
) -> RenyiMaximumCurvature:
    return RenyiMaximumCurvature(
        lower=point.lower,
        upper=point.upper,
        additive_tolerance=tolerance.lower(),
        p=p,
        delta=delta,
        alpha=alpha,
        best_sample_u=Fraction(0),
        point_evaluations=1,
        processed_boxes=0,
        unresolved_boxes_at_return=0,
        maximum_depth_reached=0,
        curvature_upper=arb(0),
        used_u_zero_shortcut=True,
    )


# ---------------------------------------------------------------------------
# Curvature bounds
# ---------------------------------------------------------------------------


def _curvature_bound(
    p: arb,
    delta: arb,
    m: int,
    u_max: arb,
) -> arb:
    """A p-dependent upper bound for sup_{0<=u<=u_max}|D_m''(u)|.

    For posterior probabilities

        v = p exp(X)/(1-p+p exp(X)),
        w = p exp(Y)/(1-p+p exp(Y)),

    Price differentiation gives

        M'  = Delta^2 E[f P1(v,w)],
        M'' = Delta^4 E[f P2(v,w)],

    where f=A^m B^(1-m), P1=m(m-1)(t^2-vw), and P2 is an
    explicit degree-four polynomial. Every positive monomial expectation
    E[f v^i w^j] using B>=1-p and B>=p exp(Y), and the smaller bound is retained. 
    All remaining Gaussian moments are finite sums available in closed form.
    """
    # The curvature formula applies only to 0<p<1. The bound is evaluated on [0,u_max],
    # where u_max is 1/2 globally or the right endpoint of an adaptive subinterval.
    if m < 2:
        raise ValueError("integer curvature bound requires m >= 2")
    q = 1 - p
    if p.lower() <= 0 or q.lower() <= 0:
        raise ValueError("p-dependent curvature bound requires 0 < p < 1")
    if u_max.lower() < 0 or u_max.upper() > arb("1/2"):
        raise ValueError("u_max must lie in [0,1/2]")

    d2 = delta * delta

    # Bound E[A^m B^(1-m) v^i w^j], where v and w are defined as above. For each k, the code derives one
    # bound from B>=1-p and another from B>=p*exp(Y), then keeps the smaller.
    # The exponential factors are maximized over 0<=u<=u_max.
    def monomial_upper(i: int, j: int) -> arb:
        if i < 0 or j < 0 or i > m:
            raise ValueError("invalid posterior monomial")

        total = arb(0)

        for k in range(m - i + 1):
            q_factor = (
                comb(m - i, k)
                * q ** (1 - i - j - k)
                * p ** (i + k + j)
            )
            q_exponent = (
                (i + k) * (i + k - 1)
                + 2 * u_max * j * (i + j + k - 1)
            ) * d2 / 2

            p_factor = (
                comb(m - i, k)
                * q ** (m - i - k)
                * p ** (1 + k + i - m)
            )
            p_exponent = (
                (i + k) * (i + k - 1)
                + 2 * u_max * (m - 1) * (m - i - k)
            ) * d2 / 2

            using_q = (q_factor * q_exponent.exp()).upper()
            using_p = (p_factor * p_exponent.exp()).upper()

            total += using_q if using_q <= using_p else using_p

        return total.upper()

    # Only the eight posterior monomials appearing in the first and second
    # derivatives are needed.
    keys = (
        (0, 2),
        (1, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
    )
    U = {key: monomial_upper(*key) for key in keys}

    # r1 bounds |M'(u)|/Delta^2 by applying absolute values to
    # m(m-1)(w^2-v*w).
    c = arb(m * (m - 1))
    r1 = (c * (U[(0, 2)] + U[(1, 1)])).upper()

    # r2 bounds |M''(u)|/Delta^4 using the displayed absolute coefficients of
    # the exact degree-four posterior polynomial.
    # Coefficients of P2:
    #   2c w^2 - 4c(m+1)w^3 + c(m+1)(m+2)w^4
    #   - c vw + 5mc vw^2 - 2mc(m+1)vw^3
    #   - m(m-1)^2 v^2w + m^2(m-1)^2 v^2w^2.
    coefficients = {
        (0, 2): 2 * m * (m - 1),
        (0, 3): 4 * m * (m - 1) * (m + 1),
        (0, 4): m * (m - 1) * (m + 1) * (m + 2),
        (1, 1): m * (m - 1),
        (1, 2): 5 * m * m * (m - 1),
        (1, 3): 2 * m * m * (m - 1) * (m + 1),
        (2, 1): m * (m - 1) * (m - 1),
        (2, 2): m * m * (m - 1) * (m - 1),
    }
    r2 = arb(0)
    for key, coefficient in coefficients.items():
        r2 += coefficient * U[key]
    r2 = r2.upper()

    # Since D=log(M)/(m-1),
    # |D''| <= Delta^4*(|M''|/M + |M'|^2/M^2)/(m-1).
    density_curvature = (d2 * d2 * (r2 + r1 * r1) / (m - 1)).upper()

    # After normalizing by M, every posterior monomial is also at most one;
    # take the smaller of this bound and the p-dependent bound above.
    one = arb(1)
    normalized_U = {
        key: value if value <= one else one
        for key, value in U.items()
    }
    normalized_r1 = (
        c * (normalized_U[(0, 2)] + normalized_U[(1, 1)])
    ).upper()
    normalized_r2 = arb(0)
    for key, coefficient in coefficients.items():
        normalized_r2 += coefficient * normalized_U[key]
    normalized_r2 = normalized_r2.upper()
    unit_monomial_curvature = (
        d2
        * d2
        * (normalized_r2 + normalized_r1 * normalized_r1)
        / (m - 1)
    ).upper()

    return (
        unit_monomial_curvature
        if unit_monomial_curvature <= density_curvature
        else density_curvature
    ).upper()




# ---------------------------------------------------------------------------
# Interpolation upper bound on one adaptive grid interval
# ---------------------------------------------------------------------------


def _quadratic_envelope_upper(
    left_upper: arb,
    right_upper: arb,
    width: Fraction,
    curvature_upper: arb,
) -> arb:
    """Upper-bound a C^2 function between endpoint upper points.

    If |f''|<=K on [a,b], h=b-a, and t=(u-a)/h, then

        f(u) <= (1-t) f(a) + t f(b) + (K h^2/2)t(1-t).

    """
    ua = left_upper.upper()
    ub = right_upper.upper()
    h = arb(_fraction_text(width))
    C = (curvature_upper.upper() * h * h / 2).upper()
    if C.is_zero():
        return _max_upper(ua, ub)

    d = ub - ua

    if d.lower() >= C.upper():
        return ub
    if d.upper() <= -C.upper():
        return ua

    return (ua + (d + C) * (d + C) / (4 * C)).upper()


# ---------------------------------------------------------------------------
# Public adaptive maximizer
# ---------------------------------------------------------------------------


def max_renyi(
    p: Scalar,
    delta: Scalar,
    alpha: Any,
    gamma: Scalar = "1e-10",
    *,
    initial_dps: int | None = None,
    max_dps: int = 20000,
    max_cutoff: int = 10000,
    max_boxes: int = 200000,
    max_depth: int = 100,
    point_options: dict[str, Any] | None = None,
) -> RenyiMaximumCurvature:
    """Calculate max_{0<=u<=1/2} D_alpha(u) by an adaptive curvature grid.
    """
    if max_dps < 80:
        raise ValueError("max_dps must be at least 80")
    if max_boxes <= 0 or max_depth <= 0:
        raise ValueError("max_boxes and max_depth must be positive")

    # Use one precision for setup and exact comparisons. Each point evaluator
    # may independently increase precision further, but restore_precision keeps
    # those changes local to the call.
    setup_dps = min(max_dps, max(80, int(initial_dps or 80)))
    with restore_precision():
        ctx.dps = setup_dps
        integer_order = _integer_order(alpha)

        pp = _arb_input(p)
        dd = _arb_input(delta)
        aa = arb(integer_order)
        ee = _arb_input(gamma) / 2

        if pp.lower() < 0 or pp.upper() > 1:
            raise ValueError("p must lie in [0,1]")
        if dd.lower() < 0:
            raise ValueError("Delta must be nonnegative")
        if ee.lower() <= 0:
            raise ValueError("gamma must be strictly positive")

        # p=0 or Delta=0 makes the two Gaussian mixtures equal. p=1 leaves two Gaussians at
        # fixed distance Delta, so their Renyi divergence is
        # alpha*Delta^2/2 for every u.
        if pp.is_zero() or dd.is_zero():
            return _closed_form_result(arb(0), pp, dd, aa, ee)
        if (pp - 1).is_zero():
            return _closed_form_result(aa * dd * dd / 2, pp, dd, aa, ee)

        if pp.lower() <= 0 or pp.upper() >= 1:
            raise ValueError(
                "for non-boundary computation p must be supplied precisely "
                "enough that its Arb ball lies strictly inside (0,1)"
            )

        # Forward precision/cutoff limits to every pointwise integral. Copying
        # the options prevents the caller's dictionary from being mutated.
        point_options_local = dict(point_options or {})
        point_options_local.setdefault("initial_dps", setup_dps)
        point_options_local.setdefault("max_dps", int(max_dps))
        point_options_local.setdefault("max_cutoff", int(max_cutoff))

        # Try the exact endpoint theorem first.
        if _u_zero_endpoint_condition_holds(pp, dd, integer_order):
            point = renyi(
                p,
                delta,
                integer_order,
                "0",
                gamma,
                **point_options_local,
            )
            return _u_zero_result(point, pp, dd, aa, ee)

        # The global curvature bound controls every unsampled point. Local
        # bounds on [0,right] are also cached because they decrease substantially
        # near u=0 and make the interval envelopes sharper.
        curvature = _curvature_bound(
            pp, dd, integer_order, arb("1/2")
        )
        local_curvature_cache: dict[Fraction, arb] = {}

        def interval_curvature(right: Fraction) -> arb:
            """Bound curvature on [0,right], cached at dyadic endpoints."""
            cached = local_curvature_cache.get(right)
            if cached is not None:
                return cached
            local = _curvature_bound(
                pp, dd, integer_order, arb(_fraction_text(right))
            )
            # The global bound is independently valid; intersecting the two
            # protects against any avoidable dependency inflation.
            local = local if local <= curvature else curvature
            local_curvature_cache[right] = local.upper()
            return local_curvature_cache[right]

        # The adaptive algorithm does not
        # infer the final error from this allocation: it terminates only when
        # its global upper minus its sampled lower is itself at most gamma.
        gamma_fraction = _exact_arb_fraction(ee.lower())
        point_tolerance = gamma_fraction / 2

        # Cache point at exact Fraction locations. Adaptive
        # bisection therefore reuses a shared endpoint only inside this one
        # maximum calculation.
        cache: dict[Fraction, RenyiInterval] = {}
        point_evaluations = 0
        processed_boxes = 0
        maximum_depth_reached = 0
        best_lower = arb(0)
        best_u = Fraction(0)

        def evaluate(u_value: Fraction) -> RenyiInterval:
            nonlocal point_evaluations, best_lower, best_u
            cached = cache.get(u_value)
            if cached is not None:
                return cached
            result = renyi(
                p,
                delta,
                integer_order,
                _fraction_text(u_value),
                _fraction_text(point_tolerance),
                **point_options_local,
            )
            cache[u_value] = result
            point_evaluations += 1
            # The largest pointwise lower endpoint is a valid lower bound on the
            # global maximum. Record the exact sampled u attaining it.
            if result.lower > best_lower:
                best_lower = result.lower
                best_u = u_value
            return result

        # Both endpoints are needed before the first interval envelope can be
        # formed; either endpoint may already provide the best lower bound.
        evaluate(Fraction(0))
        evaluate(Fraction(1, 2))

        heap: list[tuple[Fraction, int, _Box]] = []
        serial = 0

        # For [left,right], combine endpoint upper bounds with a curvature
        # bound to cover every unsampled u in the interval. Intervals whose
        # entire envelope is below the current lower bound cannot contain a
        # better maximum and are discarded.
        def push_interval(left: Fraction, right: Fraction, depth: int) -> None:
            nonlocal serial, processed_boxes, maximum_depth_reached
            processed_boxes += 1
            maximum_depth_reached = max(maximum_depth_reached, depth)
            left_value = evaluate(left)
            right_value = evaluate(right)
            upper = _quadratic_envelope_upper(
                left_value.upper,
                right_value.upper,
                right - left,
                interval_curvature(right),
            )
            if upper <= best_lower:
                return

            box = _Box(left=left, right=right, upper=upper, depth=depth)
            key = -_exact_arb_fraction(upper.upper())
            heappush(heap, (key, serial, box))
            serial += 1

        push_interval(Fraction(0), Fraction(1, 2), 0)

        # A newly improved sampled lower bound can dominate boxes already in the
        # heap; remove them before computing the current global gap.
        def discard_dominated() -> None:
            while heap and heap[0][2].upper <= best_lower:
                heappop(heap)

        # The maximum of the sampled lower values is a global lower bound. Every
        # point is either in a surviving box or in a discarded box whose envelope
        # was already at most that lower bound. Hence the largest surviving box
        # envelope, or the lower bound if none survives, is a global upper bound.
        # Their difference is used as a stopping criterion.
        while True:
            discard_dominated()
            global_upper = best_lower.upper() if not heap else heap[0][2].upper.upper()
            if global_upper < best_lower:
                global_upper = best_lower.upper()
            gap = (global_upper - best_lower).upper()
            if gap <= ee.lower():
                return RenyiMaximumCurvature(
                    lower=best_lower.lower(),
                    upper=global_upper.upper(),
                    additive_tolerance=ee.lower(),
                    p=pp,
                    delta=dd,
                    alpha=aa,
                    best_sample_u=best_u,
                    point_evaluations=point_evaluations,
                    processed_boxes=processed_boxes,
                    unresolved_boxes_at_return=len(heap),
                    maximum_depth_reached=maximum_depth_reached,
                    curvature_upper=curvature,
                )

            if not heap:
                raise RuntimeError(
                    "internal coverage error: no unresolved boxes remain but "
                    f"the gap is {gap}"
                )
            if processed_boxes >= max_boxes:
                raise RuntimeError(
                    "max_boxes was reached; "
                    f"last lower={best_lower}, upper={global_upper}, gap={gap}"
                )

            # Bisect only the worst unresolved box, calculate its midpoint, and
            # replace it by two curvature-covered intervals. At every iteration,
            # each point is either covered by a surviving interval or has already
            # been guaranteed to be unable to beat the global lower bound.
            _, _, box = heappop(heap)
            if box.upper <= best_lower:
                continue
            if box.depth >= max_depth:
                raise RuntimeError(
                    "max_depth was reached; "
                    f"box=[{box.left},{box.right}], gap={gap}"
                )

            midpoint = (box.left + box.right) / 2
            evaluate(midpoint)
            push_interval(box.left, midpoint, box.depth + 1)
            push_interval(midpoint, box.right, box.depth + 1)