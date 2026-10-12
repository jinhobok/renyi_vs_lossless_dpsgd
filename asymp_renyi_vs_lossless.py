#!/usr/bin/env python3
"""Compare finite-T privacy curves with their Tp^2 -> nu limits."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})
import numpy as np
from scipy.special import log_ndtr, ndtr
from matplotlib.ticker import MaxNLocator

from prv import FixedSizeReplaceOneGaussianPRV
from isosceles_rdp_scipy import estimate_max_renyi


def comma_list(text, cast=float):
    return [cast(value) for value in text.split(",")]


def prv_limit_log_delta(nu, sigma, epsilon):
    mu = math.sqrt(
        2.0
        * nu
        * (
            math.exp(4.0 / sigma**2) * ndtr(3.0 / sigma)
            + 3.0 * ndtr(-1.0 / sigma)
            - 2.0
        )
    )
    first = float(log_ndtr(-epsilon / mu + mu / 2.0))
    second = float(epsilon + log_ndtr(-epsilon / mu - mu / 2.0))
    gap = second - first
    if gap >= 0.0:
        return 0.0
    return min(0.0, first + math.log(-math.expm1(gap)))


def isosceles_asymptotic_bound_log_delta(nu, sigma, epsilon):
    lam = 2.0 * nu * (math.exp(4.0 / sigma**2) - math.exp(2.0 / sigma**2))
    value = (
        math.log(lam / (lam + epsilon))
        + (epsilon / lam) * math.log(epsilon / (lam + epsilon))
        - epsilon**2 / (2.0 * lam)
        + epsilon / 2.0
    )

    return min(0.0, value)


def isosceles_log_deltas(p, sigma, steps, epsilons, alpha_max, accuracy):
    orders = np.arange(2, alpha_max + 1, dtype=float)
    rdp = []
    for alpha in orders.astype(int):
        result = estimate_max_renyi(p, 2.0 / sigma, alpha, accuracy=accuracy)
        if not result.converged:
            raise RuntimeError(f"isosceles calculation did not converge at alpha={alpha}")
        rdp.append(result.value)
    rdp = np.asarray(rdp)
    values = (
        -np.log(orders)[:, None]
        + (orders[:, None] - 1.0)
        * (
            steps * rdp[:, None]
            - np.asarray(epsilons)[None, :]
            + np.log1p(-1.0 / orders)[:, None]
        )
    )
    return np.minimum(0.0, np.min(values, axis=0))

def scientific_tick(x, pos):
    if x == 0:
        return r"$0$"
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / 10**exponent
    return fr"${coefficient:g}\times 10^{{{exponent}}}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=2)
    parser.add_argument("--steps", default="2000,4000,8000,16000,32000")
    parser.add_argument("--epsilons", default="2,3,4")
    parser.add_argument("--alpha-max", type=int, default=64)
    parser.add_argument("--scipy-accuracy", type=float, default=1e-6)
    parser.add_argument("--prv-epsilon-error", type=float, default=0.01)
    parser.add_argument("--prv-delta-error", type=float, default=1e-12)
    parser.add_argument("--output", default="asymp_renyi_vs_lossless.pdf")
    args = parser.parse_args()

    from prv_accountant import PRVAccountant

    steps = sorted(set(comma_list(args.steps, int)))
    epsilons = comma_list(args.epsilons)
    if args.nu <= 0 or args.sigma <= 0 or args.alpha_max < 2:
        raise ValueError("nu and sigma must be positive, and alpha-max must be at least 2")
    if any(epsilon <= 0.0 for epsilon in epsilons):
        raise ValueError("epsilons must be positive")
    if any(t <= args.nu for t in steps):
        raise ValueError("every T must exceed nu so that p=sqrt(nu/T) lies in (0,1)")

    prv_logs = {epsilon: [] for epsilon in epsilons}
    iso_logs = {epsilon: [] for epsilon in epsilons}

    for steps_value in steps:
        p = math.sqrt(args.nu / steps_value)
        accountant = PRVAccountant(
            prvs=[FixedSizeReplaceOneGaussianPRV(p, args.sigma)],
            max_self_compositions=[steps_value],
            eps_error=args.prv_epsilon_error,
            delta_error=args.prv_delta_error,
        )
        composed = accountant.compute_composition([steps_value])
        for epsilon in epsilons:
            delta = min(1.0, max(0.0, composed.compute_delta_estimate(epsilon)))
            prv_logs[epsilon].append(math.log(max(delta, np.finfo(float).tiny)))

        values = isosceles_log_deltas(
            p,
            args.sigma,
            steps_value,
            epsilons,
            args.alpha_max,
            args.scipy_accuracy,
        )
        for epsilon, value in zip(epsilons, values):
            iso_logs[epsilon].append(float(value))

    green = np.array([27, 158, 119]) / 255
    purple = np.array([117, 112, 179]) / 255

    plt.figure(figsize=(6, 3.75))
    for index, epsilon in enumerate(epsilons):
        factor = 0.7 ** (index - 0.5)
        green_color = green * factor
        purple_color = purple * factor
        plt.plot(steps, iso_logs[epsilon], "o-", color=green_color, label=fr"Our bounds, $\varepsilon={epsilon:g}$")
        plt.plot(
                    steps,
                    [isosceles_asymptotic_bound_log_delta(args.nu, args.sigma, epsilon)] * len(steps),
                    "--",
                    color=green_color,
                    label=fr"Limit (ours), $\varepsilon={epsilon:g}$",
                )
        plt.plot(steps, prv_logs[epsilon], "^-", color=purple_color, label=fr"Lossless accountant, $\varepsilon={epsilon:g}$")
        plt.plot(
            steps,
            [prv_limit_log_delta(args.nu, args.sigma, epsilon)] * len(steps),
            ":",
            color=purple_color,
            label=fr"Limit (lossless), $\varepsilon={epsilon:g}$",
        )
        

    plt.xscale("log")
    plt.xticks(steps, [str(steps_value) for steps_value in steps])
    plt.xlabel(r"Iterations $T$, with $p=\sqrt{\nu/T}$")
    plt.ylabel(r"Privacy $\log\delta$")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(fr"$Tp^2 = \nu={args.nu:g}$, $\sigma={args.sigma:g}$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8, ncol=len(epsilons), loc="upper center", bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
