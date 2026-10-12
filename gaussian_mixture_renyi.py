"""Pointwise Rényi divergence for integer orders.

For an integer order ``alpha >= 2`` this module rigorously calculates

    D_alpha(P_a || P_b(u)) = log(M_alpha(p, Delta, u)) / (alpha - 1),

where

    P_a    = (1-p) N(0,I_2) + p N(a,I_2),
    P_b(u) = (1-p) N(0,I_2) + p N(b(u),I_2),
    a      = Delta e_1,
    b(u)   = Delta (u e_1 + sqrt(2u-u^2) e_2),
    0 <= u <= 1/2.

The exact two-dimensional Gaussian moment is reduced to
one-dimensional Arb integral. Both infinite Gaussian tails are bounded
analytically and added to the integral over a finite interval. The returned upper endpoint
is an upper bound with additive error <= gamma.

Only integer Rényi orders are implemented in this file.

Dependency: ``pip install python-flint``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

from flint import acb, acb_poly, arb, ctx

from utils import (
    Scalar,
    arb_input as _arb_input,
    integer_order as _integer_order,
    lower_text as _lower_text,
    restore_precision,
    upper_text as _upper_text,
    upper_text_with_tolerance,
)


@dataclass(frozen=True)
class RenyiInterval:
    """Certified intervals for the moment and the Rényi divergence."""

    lower: arb
    upper: arb
    moment_lower: arb
    moment_upper: arb
    alpha: arb
    u: arb
    cutoff_left: arb
    cutoff_right: arb
    decimal_precision: int
    method: str
    additive_tolerance: arb
    central_moment_lower: arb
    central_moment_upper: arb
    left_tail_upper: arb
    right_tail_upper: arb

    @property
    def width(self) -> arb:
        return (self.upper - self.lower).upper()

    @property
    def moment_width(self) -> arb:
        return (self.moment_upper - self.moment_lower).upper()

    @property
    def total_tail_upper(self) -> arb:
        return (self.left_tail_upper + self.right_tail_upper).upper()

    def upper_string(self, digits: int = 30) -> str:
        """Outward-rounded fixed-point decimal upper bound."""
        return upper_text_with_tolerance(
            self.upper, self.lower, self.additive_tolerance, digits
        )

    def lower_string(self, digits: int = 30) -> str:
        return _lower_text(self.lower, digits)

    def width_string(self, digits: int = 30) -> str:
        return _upper_text(self.width, digits)

    def moment_upper_string(self, digits: int = 30) -> str:
        return _upper_text(self.moment_upper, digits)

    def moment_lower_string(self, digits: int = 30) -> str:
        return _lower_text(self.moment_lower, digits)


def _phi_cdf(x: arb) -> arb:
    return (-x / arb(2).sqrt()).erfc() / 2


# Each tail term has two independently valid bounds, obtained from the two
# denominator lower bounds used later. We only keep the smaller upper bound.
def _min_of_upper_bounds(x: arb, y: arb) -> arb:
    xu = x.upper()
    yu = y.upper()
    return xu if xu <= yu else yu


def _validate_common(p: arb, delta: arb, u: arb, gamma: arb) -> None:
    if gamma.lower() <= 0:
        raise ValueError("gamma must be strictly positive")
    if delta.lower() < 0:
        raise ValueError("Delta must be nonnegative")
    if u.lower() < 0 or u.upper() > arb("1/2").upper():
        raise ValueError("u must lie in [0,1/2]")
    if p.lower() < 0 or p.upper() > 1:
        raise ValueError("p must lie in [0,1]")


# If the moment is known within relative factor
# exp((alpha-1)*d_tolerance), then log(moment)/(alpha-1) is known within
# d_tolerance. expm1 computes that relative moment gap.
def _relative_moment_budget(alpha: arb, d_tolerance: arb) -> arb:
    return (((alpha - 1).lower() * d_tolerance.lower()).lower()).expm1().lower()


# Allocate only a small part of the relative moment budget to each adaptive
# quadrature call. The explicit cap avoids asking Arb for an unnecessarily
# loose tolerance when the requested final gamma is large.
def _quadrature_tolerance(relative_budget: arb, divisor: int) -> arb:
    tol = (relative_budget / divisor).lower()
    return arb("1e-4").lower() if tol > arb("1e-4") else tol


# Convert moment endpoints monotonically through log(M)/(alpha-1).
def _renyi_endpoints(moment_lower: arb, moment_upper: arb, alpha: arb) -> tuple[arb, arb]:
    denominator = alpha - 1
    lower = ((moment_lower - 1).log1p() / denominator).lower()
    upper = ((moment_upper - 1).log1p() / denominator).upper()
    return lower, upper


def _make_result(
    *,
    moment_lower: arb,
    moment_upper: arb,
    alpha: arb,
    u: arb,
    cutoff_left: arb,
    cutoff_right: arb,
    decimal_precision: int,
    method: str,
    gamma: arb,
    central_moment_lower: arb,
    central_moment_upper: arb,
    left_tail_upper: arb,
    right_tail_upper: arb,
    direct_d: arb | None = None,
) -> RenyiInterval:
    d_lower, d_upper = _renyi_endpoints(moment_lower, moment_upper, alpha)
    if direct_d is not None:
        direct_lower = direct_d.lower()
        direct_upper = direct_d.upper()
        if direct_lower > d_lower:
            d_lower = direct_lower
        if direct_upper < d_upper:
            d_upper = direct_upper
    return RenyiInterval(
        lower=d_lower,
        upper=d_upper,
        moment_lower=moment_lower.lower(),
        moment_upper=moment_upper.upper(),
        alpha=alpha,
        u=u,
        cutoff_left=cutoff_left,
        cutoff_right=cutoff_right,
        decimal_precision=decimal_precision,
        method=method,
        additive_tolerance=gamma.lower(),
        central_moment_lower=central_moment_lower,
        central_moment_upper=central_moment_upper,
        left_tail_upper=left_tail_upper,
        right_tail_upper=right_tail_upper,
    )


def _boundary_result(
    p: arb,
    delta: arb,
    alpha: arb,
    u: arb,
    gamma: arb,
    dps: int,
    method: str,
) -> RenyiInterval | None:
    # If p=0 or Delta=0, the two mixtures coincide and D_alpha=0. If p=1, D_alpha=alpha*Delta^2/2.
    if p.is_zero() or delta.is_zero():
        moment, direct_d = arb(1), arb(0)
    elif (p - 1).is_zero():
        direct_d = alpha * delta * delta / 2
        moment = (direct_d * (alpha - 1)).exp()
    else:
        return None
    return _make_result(
        moment_lower=moment.lower(),
        moment_upper=moment.upper(),
        alpha=alpha,
        u=u,
        cutoff_left=arb(0),
        cutoff_right=arb(0),
        decimal_precision=dps,
        method=method,
        gamma=gamma,
        central_moment_lower=moment.lower(),
        central_moment_upper=moment.upper(),
        left_tail_upper=arb(0),
        right_tail_upper=arb(0),
        direct_d=direct_d,
    )


def _piecewise_integral(
    integrand: Any,
    left: int,
    right: int,
    abs_tol: arb,
    rel_tol: arb,
    max_piece_width: int = 8,
) -> acb:
    """Finite integral split into moderate-width pieces."""
    # Splitting a long finite interval keeps Arb's adaptive integrator from
    # spending one global error budget on a potentially nonuniform integrand. The
    # absolute tolerance is divided among pieces, so summing the 
    # piece intervals preserves the requested total absolute allowance.
    points = list(range(left, right, max_piece_width)) + [right]
    local_abs_tol = (abs_tol / (len(points) - 1)).lower()
    total = acb(0)
    for start, stop in zip(points[:-1], points[1:]):
        total += acb.integral(
            integrand,
            start,
            stop,
            abs_tol=local_abs_tol,
            rel_tol=rel_tol,
            use_heap=True,
        )
    return total


def _renyi_integer_impl(
    p: Scalar,
    delta: Scalar,
    alpha: int,
    u: Scalar,
    gamma: Scalar = "1e-12",
    *,
    initial_dps: int | None = None,
    max_dps: int = 20000,
    max_cutoff: int = 10000,
) -> RenyiInterval:
    """additive-gamma evaluation for integer alpha >= 2.

    The Arb integral is over a truncated interval, whereas the left and right tails 
    are bounded analytically and added before
    taking log and then dividing by alpha - 1.
    """
    m = _integer_order(alpha)

    if max_dps < 40:
        raise ValueError("max_dps must be at least 40")
    initial_cutoff = 8
    if max_cutoff < initial_cutoff:
        raise ValueError("max_cutoff must be at least 8")
    dps = max(40, int(initial_dps or 80))
    if dps > max_dps:
        raise ValueError("initial_dps must not exceed max_dps")
    last: RenyiInterval | None = None

    # Increase decimal precision geometrically until the final divergence
    # interval is narrow enough.
    while dps <= max_dps:
        ctx.dps = dps
        pp = _arb_input(p)
        dd = _arb_input(delta)
        uu = _arb_input(u)
        ee = _arb_input(gamma)
        aa = arb(m)
        _validate_common(pp, dd, uu, ee)
        # Use gamma/2 as a conservative internal width target.
        internal_goal = (ee.lower() / 2).lower()

        boundary = _boundary_result(
            pp, dd, aa, uu, ee, dps, "integer closed form"
        )
        if boundary is not None:
            if boundary.width <= internal_goal:
                return boundary
            dps *= 2
            continue

        qq = 1 - pp
        if pp.lower() <= 0 or qq.lower() <= 0:
            raise ValueError(
                "for non-boundary computation p must lie strictly inside (0,1)"
            )

        # Translate the desired additive error in D_alpha into an allowed
        # relative error in the positive moment, then reserve a fraction of it
        # for the quadrature calculation.
        relative_budget = _relative_moment_budget(aa, internal_goal)
        if relative_budget.lower() <= 0:
            dps *= 2
            continue
        quad_tol = _quadrature_tolerance(relative_budget, 128)

        # The second mean has norm beta=Delta*sqrt(2u).
        d2 = dd * dd
        beta2 = 2 * d2 * uu
        beta = beta2.sqrt()

        # At u=0, the second mixture is exactly N(0,I), so B=1. Expanding
        # ((1-p)+p exp(X))^m and using
        # E[exp(kX)]=exp(k(k-1)Delta^2/2) gives the finite sum below.
        if beta.is_zero():
            moment = arb(0)
            for k in range(m + 1):
                moment += (
                    comb(m, k)
                    * qq ** (m - k)
                    * pp**k
                    * (k * (k - 1) * d2 / 2).exp() 
                )
            result = _make_result(
                moment_lower=arb(1) if moment.lower() < 1 else moment.lower(),
                moment_upper=moment.upper(),
                alpha=aa,
                u=uu,
                cutoff_left=arb(0),
                cutoff_right=arb(0),
                decimal_precision=dps,
                method="integer closed form (u=0)",
                gamma=ee,
                central_moment_lower=moment.lower(),
                central_moment_upper=moment.upper(),
                left_tail_upper=arb(0),
                right_tail_upper=arb(0),
            )
            last = result
            if result.width <= internal_goal:
                return result
            dps *= 2
            continue

        # For Z~N(0,I), divide both Gaussian-mixture densities by the density of
        # Z. The first ratio is
        #   A(Z)=(1-p)+p exp(Delta*Z_1-Delta^2/2),
        # and the second is
        #   B_u(Z)=(1-p)+p exp(b(u).Z-|b(u)|^2/2),
        # where b(u)=Delta*(u,sqrt(2u-u^2)). Thus the order-m moment is
        # E[A(Z)^m/B_u(Z)^(m-1)].
        #
        # Rotate Z so V is in the direction of b(u), and let W be the independent
        # orthogonal coordinate. Since beta=Delta*sqrt(2u), conditional on V=v,
        #   log((B_u-(1-p))/p) = beta*v-beta^2/2,
        #   log((A-(1-p))/p)   = beta*v/2 + sqrt(Delta^2-beta^2/4)*W-Delta^2/2.
        # Expanding A^m and integrating W leaves one integral
        # over the single standard-normal variable V.
        q_tail_coeff: list[arb] = []
        p_tail_coeff: list[arb] = []
        lambda_q: list[arb] = []
        lambda_p: list[arb] = []
        central_coeff: list[arb] = []

        # For the k-th term in the expansion of A^m, B_u^(1-m) is bounded in
        # two independent ways: B_u>=1-p and B_u>=p exp(beta*V-beta^2/2).
        # After completing the Gaussian square, the first bound has coefficient
        #   binom(m,k)(1-p)^(1-k)p^k exp(k(k-1)Delta^2/2)
        # and normal mean shift k*beta/2. The second has coefficient
        #   binom(m,k)(1-p)^(m-k)p^(k-m+1)
        #   * exp(k(k-1)Delta^2/2+(m-1)(m-k)beta^2/2)
        # and normal mean shift beta*(k/2-(m-1)). The code keeps the smaller bound for each
        # k. central_coeff separately stores the coefficients of
        # t^k, t=exp(beta*V/2), in the conditional numerator polynomial.
        for k in range(m + 1):
            kk = arb(k)
            q_tail_coeff.append(
                comb(m, k) * qq ** (1 - k) * pp**k * (k * (k - 1) * d2 / 2).exp()
            )
            p_tail_coeff.append(
                comb(m, k)
                * qq ** (m - k)
                * pp ** (k - m + 1)
                * (
                    k * (k - 1) * d2 / 2
                    + (m - 1) * (m - k) * beta2 / 2
                ).exp()
            )
            lambda_q.append(kk * beta / 2)
            lambda_p.append(beta * (kk / 2 - (m - 1)))
            central_coeff.append(
                comb(m, k)
                * qq ** (m - k)
                * pp**k
                * (
                    k * (k - 1) * d2 / 2
                    - k * k * beta2 / 8
                ).exp()
            )

        # Integrating each exponential term against the standard normal density
        # outside V<-cutoff gives a shifted Phi tail.
        def f_left_tail_upper(cutoff: int) -> arb:
            """Upper bound for the left tail of F=A^m B^(1-m)."""
            x = arb(cutoff)
            total = arb(0)
            for k in range(m + 1):
                using_q = q_tail_coeff[k] * _phi_cdf(-x - lambda_q[k])
                using_p = p_tail_coeff[k] * _phi_cdf(-x - lambda_p[k])
                total += _min_of_upper_bounds(using_q, using_p)
            return total.upper()

        # The right-tail formula is the analogous integral over V>cutoff.
        def f_right_tail_upper(cutoff: int) -> arb:
            """Upper bound for the right tail of F=A^m B^(1-m)."""
            x = arb(cutoff)
            total = arb(0)
            for k in range(m + 1):
                using_q = q_tail_coeff[k] * _phi_cdf(lambda_q[k] - x)
                using_p = p_tail_coeff[k] * _phi_cdf(lambda_p[k] - x)
                total += _min_of_upper_bounds(using_q, using_p)
            return total.upper()

        # The exact conditional moment is a degree-m numerator polynomial in
        # t=exp(beta*V/2), divided by
        #   ((1-p)+p exp(-beta^2/2)t^2)^(m-1).
        # More precisely, central_coeff defines N(t) such that
        #   E_W[A^m | V=v]/B_u(v)^(m-1)
        #     = N(t)/((1-p)+p exp(-beta^2/2)t^2)^(m-1).
        inv_sqrt_2pi = 1 / (2 * arb.pi()).sqrt()

        exp_minus_beta2_over_2 = (-beta2 / 2).exp()
        b_factor = pp * exp_minus_beta2_over_2

        # Evaluate the degree-m numerator in FLINT's compiled polynomial code
        moment_polynomial = acb_poly(central_coeff)

        # Multiplying the conditional moment by the standard normal density of V
        # gives the full one-dimensional integrand.
        def integrand(v: acb, _analytic: bool) -> acb:
            t = (beta * v / 2).exp()
            numerator = moment_polynomial(t)
            B = qq + b_factor * t * t
            conditional_moment = numerator / (B ** (m - 1))
            return inv_sqrt_2pi * (-v * v / 2).exp() * conditional_moment

        # Start from [-8,8]. Cutoffs are enlarged only when the analytic tail
        # bounds are too large relative to the current certified moment lower bound.
        L = initial_cutoff
        R = initial_cutoff

        # First integrate a moderate truncated interval. Its rigorous lower mass
        # then provides the scale for relative tail truncation. Enlarging the
        # interval can only improve this lower mass.
        while True:
            left = f_left_tail_upper(L)
            right = f_right_tail_upper(R)
            central = _piecewise_integral(
                integrand,
                -L,
                R,
                abs_tol=quad_tol,
                rel_tol=quad_tol,
            )
            if not central.is_finite() or not central.imag.contains(0):
                break

            # The truncated integral is nonnegative. Intersect its
            # numerical real interval with zero before using it as a lower bound.
            central_lower = central.real.lower()
            central_upper = central.real.upper()
            if central_upper < 0:
                break
            if central_lower < 0:
                central_lower = arb(0)

            # The integrand and the tails are nonnegative. Therefore
            # central_lower bounds the full moment from below, while
            # central_upper+left+right bounds the full moment from above. Independently,
            # Hölder's inequality gives the lower bound M_m>=1.
            moment_lower = arb(1) if central_lower < 1 else central_lower
            moment_upper = (central_upper + left + right).upper()
            result = _make_result(
                moment_lower=moment_lower,
                moment_upper=moment_upper,
                alpha=aa,
                u=uu,
                cutoff_left=arb(-L),
                cutoff_right=arb(R),
                decimal_precision=dps,
                method="integer direct one-dimensional",
                gamma=ee,
                central_moment_lower=central_lower,
                central_moment_upper=central_upper,
                left_tail_upper=left,
                right_tail_upper=right,
            )
            last = result
            if result.width <= internal_goal:
                return result

            # Use the current positive moment lower bound to turn the allowed
            # relative error into an absolute target for each tail.
            # If each tail is at most eta*M_lower/32, their combined relative
            # contribution is at most eta/16.
            side_target = (
                relative_budget.lower() * moment_lower.lower() / 32
            ).lower()
            if side_target.lower() <= 0:
                break

            changed = False
            while left > side_target:
                L += 1
                changed = True
                if L > max_cutoff:
                    raise RuntimeError("left-tail cutoff exceeded max_cutoff")
                left = f_left_tail_upper(L)
            while right > side_target:
                R += 1
                changed = True
                if R > max_cutoff:
                    raise RuntimeError("right-tail cutoff exceeded max_cutoff")
                right = f_right_tail_upper(R)
            if not changed:
                break

        # If finite precision prevents the central integral or width test from
        # succeeding, double precision and recompute every coefficient, tail,
        # and integral from the original exact inputs.
        dps *= 2

    if last is not None:
        raise RuntimeError(
            "failed to certify the requested D_alpha width before max_dps; "
            f"last certified width was {last.width}"
        )
    raise RuntimeError("failed to produce a finite certified enclosure")


def renyi(
    p: Scalar,
    delta: Scalar,
    alpha: int,
    u: Scalar,
    gamma: Scalar = "1e-12",
    **kwargs: Any,
) -> RenyiInterval:
    """Certified additive-gamma evaluation for an integer order."""
    with restore_precision():
        return _renyi_integer_impl(
            p, delta, alpha, u, gamma, **kwargs
        )


def renyi_upper_bound(
    p: Scalar, delta: Scalar, alpha: Any, u: Scalar,
    gamma: Scalar = "1e-12", **kwargs: Any,
) -> arb:
    return renyi(p, delta, alpha, u, gamma, **kwargs).upper


def renyi_plain_upper_bound(
    p: Scalar, delta: Scalar, alpha: Any, u: Scalar,
    gamma: Scalar = "1e-10", *, digits: int = 30, **kwargs: Any,
) -> str:
    return renyi(p, delta, alpha, u, gamma, **kwargs).upper_string(digits)
