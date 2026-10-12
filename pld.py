"""One step privacy loss function formulas for fixed-size replace-one DP-SGD.

Let S=N(0,1), T=N(shift,1), and M=(1-q)S+qT. The ``artificial``
dominating privacy-loss distribution is made by joining the forward PLD of
M versus S and the reverse PLD of S versus M at privacy loss zero.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, log_ndtr, logsumexp, ndtr


# Throughout this file q is the fixed sampling probability and shift=2/sigma.
def _validate(q: float, shift: float) -> tuple[float, float]:
    q = float(q)
    shift = float(shift)
    if not 0.0 < q < 1.0:
        raise ValueError("sampling_probability must lie in (0,1)")
    if shift <= 0.0:
        raise ValueError("replacement_shift must be positive")
    return q, shift


# Inverting the likelihood ratio requires calculating log(exp(a)-1+q). The direct formula
# is accurate for moderate a, while factoring out exp(a) prevents overflow and
# cancellation for large a.
def _log_expm1_plus_q(value: np.ndarray, q: float) -> np.ndarray:
    """Stable log(exp(value) - 1 + q) for nonnegative value."""
    value = np.asarray(value, dtype=float)
    result = np.empty_like(value)
    moderate = value < 40.0
    result[moderate] = np.log(np.expm1(value[moderate]) + q)
    result[~moderate] = value[~moderate] + np.log1p(
        (q - 1.0) * np.exp(-value[~moderate])
    )
    return result


def privacy_loss_threshold(
    absolute_loss: Any,
    sampling_probability: float,
    replacement_shift: float,
) -> Any:
    """Return x such that log((1-q)+q exp(shift*x-shift^2/2))=|loss|."""
    # For S=N(0,1), T=N(shift,1), and M=(1-q)S+qT, the log likelihood ratio is
    #
    #   L(x)=log(M(x)/S(x))
    #       =log((1-q)+q*exp(shift*x-shift^2/2)).
    #
    # L is strictly increasing. Solving L(x)=a for a>=0 gives exactly the
    # threshold below.
    q, shift = _validate(sampling_probability, replacement_shift)
    loss = np.asarray(absolute_loss, dtype=float)
    if np.any(loss < 0.0):
        raise ValueError("absolute_loss must be nonnegative")
    threshold = (
        _log_expm1_plus_q(loss, q) - math.log(q) + 0.5 * shift**2
    ) / shift
    return float(threshold) if np.ndim(absolute_loss) == 0 else threshold


def fixed_size_replace_prv_cdf(
    privacy_loss: Any,
    sampling_probability: float,
    replacement_shift: float,
) -> Any:
    """CDF of the artificial fixed-size replace-one privacy random variable."""
    # The artificial dominating privacy random variable uses the positive tail
    # of L(X) under X~M=(1-q)N(0,1)+qN(shift,1) and the negative tail of -L(X) under X~S=N(0,1), with the
    # remaining probability placed at privacy loss zero. Since L(shift/2)=0:
    #
    # * for y<0, F(y)=Pr_S[-L(X)<=y]=Pr_S[X>=L^{-1}(|y|)];
    # * for y>=0, F(y)=Pr_M[L(X)<=y]=Pr_M[X<=L^{-1}(y)].
    #
    # The two formulas below are these Gaussian probabilities.
    q, shift = _validate(sampling_probability, replacement_shift)
    loss = np.asarray(privacy_loss, dtype=float)
    threshold = privacy_loss_threshold(np.abs(loss), q, shift)

    negative_branch = ndtr(-threshold)
    positive_branch = (1.0 - q) * ndtr(threshold) + q * ndtr(threshold - shift)
    result = np.where(loss < 0.0, negative_branch, positive_branch)
    result = np.clip(result, 0.0, 1.0)
    return float(result) if np.ndim(privacy_loss) == 0 else result


def fixed_size_replace_privacy_profile(
    epsilon: Any,
    sampling_probability: float,
    replacement_shift: float,
) -> Any:
    """One-step hockey-stick divergence of the artificial PLD.

    The nonnegative part is evaluated in log space. The negative part
    follows from the symmetry identity

        delta(-a) = 1 - exp(-a) + exp(-a) delta(a).
    """
    # The privacy profile of a privacy random variable Y is
    #
    #   delta(epsilon)=E[(1-exp(epsilon-Y))_+].
    #
    # For epsilon>=0 only the positive, M=(1-q)N(0,1)+qN(shift,1) vs S=N(0,1) part contributes. If
    # x=L^{-1}(epsilon), direct integration gives
    #
    #   q*Phi(shift-x) - q*exp(shift*x-shift^2/2)*Phi(-x).
    #
    # The two positive quantities can be very close, so they are formed
    # in log space before their difference is taken.
    q, shift = _validate(sampling_probability, replacement_shift)
    eps = np.asarray(epsilon, dtype=float)
    abs_eps = np.abs(eps)
    threshold = privacy_loss_threshold(abs_eps, q, shift)

    log_first = math.log(q) + log_ndtr(shift - threshold)
    log_second = (
        math.log(q)
        + shift * threshold
        - 0.5 * shift**2
        + log_ndtr(-threshold)
    )
    log_ratio = np.minimum(log_second - log_first, 0.0)
    positive = np.exp(log_first) * (-np.expm1(log_ratio))
    positive = np.clip(positive, 0.0, 1.0)

    # For negative epsilon, use delta(-a)=1-exp(-a)+exp(-a)*delta(a), with a=|epsilon|.
    result = np.where(
        eps < 0.0,
        -np.expm1(eps) + np.exp(eps) * positive,
        positive,
    )
    result = np.clip(result, 0.0, 1.0)
    return float(result) if np.ndim(epsilon) == 0 else result


# Evaluate L(x)=log(M(x)/S(x)) by logaddexp.
def _log_likelihood_ratio(x: float, q: float, shift: float) -> float:
    return float(
        np.logaddexp(
            math.log1p(-q),
            math.log(q) + shift * x - 0.5 * shift**2,
        )
    )


def _artificial_prv_first_moment(
    q: float,
    shift: float,
    upper_x: float,
) -> float:
    # x0=shift/2 is the unique point where L(x0)=0. Combining the positive
    # contribution under M=(1-q)N(0,1)+qN(shift,1) and the negative contribution under S=N(0,1) shows that the
    # artificial PRV's first moment over losses up to L(upper_x) is
    #
    #   integral_{x0}^{upper_x} L(x) * (M(x)-S(x)) dx.
    #
    x0 = shift / 2.0
    normalizer = math.sqrt(2.0 * math.pi)

    def normal_density(x: float) -> float:
        return math.exp(-0.5 * x * x) / normalizer

    def integrand(x: float) -> float:
        loss = _log_likelihood_ratio(x, q, shift)
        density_s = normal_density(x)
        density_m = (1.0 - q) * normal_density(x) + q * normal_density(x - shift)
        return loss * (density_m - density_s)

    value, _ = quad(
        integrand,
        x0,
        upper_x,
        epsabs=1e-13,
        epsrel=1e-12,
        limit=500,
    )
    return float(value)


def truncated_prv_conditional_mean(q: float, shift: float, bound: float) -> float:
    """Mean of the artificial PRV conditioned on ``abs(Y) <= bound``."""
    q, shift = _validate(q, shift)
    bound = float(bound)
    if bound <= 0.0:
        raise ValueError("bound must be positive")

    # The event |Y|<=bound corresponds to x running from shift/2 to the solution
    # of L(x)=bound on both joined branches. Divide the truncated first moment
    # by its exact CDF mass to obtain the conditional mean used as the grid mean.
    upper_x = privacy_loss_threshold(bound, q, shift)
    mass = fixed_size_replace_prv_cdf(bound, q, shift) - fixed_size_replace_prv_cdf(
        -bound, q, shift
    )
    if mass <= 0.0:
        raise RuntimeError("truncation interval has zero probability mass")
    return _artificial_prv_first_moment(q, shift, upper_x) / mass


def artificial_prv_rdp_upper(alpha: float, q: float, shift: float) -> float:
    """Integer-order RDP upper bound used to size numerical PLD domains."""
    q, shift = _validate(q, shift)
    if alpha <= 1.0:
        raise ValueError("alpha must exceed one")

    # Current callers request integer alpha. Subtracting 1e-12 before ceil snaps
    # ordinary floating-point noise just above such an integer. A noninteger
    # alpha more than 1e-12 above the previous integer is rounded to the next
    # integer, which gives an upper bound by monotonicity of Renyi divergence.
    order = max(2, int(math.ceil(float(alpha) - 1e-12)))
    x0 = shift / 2.0
    # For each k = 0, ..., order, this is the log of the k-th term obtained by
    # expanding
    #
    #     ((1-q) + q exp(shift*x - shift**2/2))**order
    #
    # and integrating that term over x >= x0 against the N(0,1) density.
    #
    # The k-th binomial term before integration is
    #
    #     binom(order,k) (1-q)**(order-k) q**k
    #         * exp(k*shift*x - k*shift**2/2).
    #
    # Completing the square in the Gaussian integral gives
    #
    #     ∫_{x0}^infty exp(k*shift*x - k*shift**2/2) phi(x) dx
    #       = exp(k*(k-1)*shift**2/2) * Phi(k*shift - x0).
    #
    # Hence the logarithm of the complete k-th contribution is
    #
    #     log binom(order,k)
    #     + (order-k) log(1-q) + k log(q)
    #     + k*(k-1)*shift**2/2
    #     + log Phi(k*shift - x0).
    #
    # The gammaln terms compute log binom(order,k), and log_ndtr computes
    # log Phi directly. The entire quantity is kept in log-space for numerical stability.
    k = np.arange(order + 1, dtype=float)
    positive_terms = (
        gammaln(order + 1.0)
        - gammaln(k + 1.0)
        - gammaln(order - k + 1.0)
        + (order - k) * math.log1p(-q)
        + k * math.log(q)
        + 0.5 * k * (k - 1.0) * shift**2
        + log_ndtr(k * shift - x0)
    )

    # To upper bound the RDP moment E[exp((order-1)Y)], split the
    # artificial PRV into its negative, zero, and positive parts.
    #
    # The positive contribution is computed exactly above.
    #
    # At Y = 0, exp((order-1)Y) = 1, so the contribution is exactly
    # the atom's probability mass.
    #
    # On the negative part Y < 0, exp((order-1)Y) <= 1. Therefore
    # its contribution to the moment is at most the total probability
    # of the negative part, Phi(-shift/2). This avoids having to
    # integrate the negative part exactly.
    #
    # Adding these three terms gives an upper bound on the RDP moment.
    zero_atom = (1.0 - q) * (2.0 * float(ndtr(x0)) - 1.0)
    log_zero = math.log(zero_atom) if zero_atom > 0.0 else -math.inf

    log_moment = float(
        logsumexp([log_zero, float(log_ndtr(-x0)), float(logsumexp(positive_terms))])
    )
    return float(np.nextafter(log_moment / (order - 1.0), math.inf))
