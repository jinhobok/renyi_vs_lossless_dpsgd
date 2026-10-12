#!/usr/bin/env python3
"""Compare one-step RDPs with their p^3 expansions."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})
from matplotlib.ticker import FuncFormatter
import numpy as np

from birrell_rdp import fswo_replace_one_rdp_profile
from isosceles_rdp_scipy import estimate_max_renyi


def comma_list(text):
    return [float(value) for value in text.split(",")]


def expansions(orders, p, sigma):
    inverse_variance = 1.0 / sigma**2
    leading = math.exp(4.0 * inverse_variance) - math.exp(2.0 * inverse_variance)
    isosceles_cubic = (
        math.exp(12.0 * inverse_variance)
        - math.exp(8.0 * inverse_variance)
        - 2.0 * math.exp(4.0 * inverse_variance)
        + 2.0 * math.exp(2.0 * inverse_variance)
    )
    birrell_cubic = math.sqrt(
        (math.exp(4.0 * inverse_variance) - 1)
        * (
            math.exp(24.0 * inverse_variance)
            - 4.0 * math.exp(12.0 * inverse_variance)
            + 6.0 * math.exp(4.0 * inverse_variance)
            - 3.0
        )
    )
    isosceles = orders * leading * p**2 - 0.5 * orders * isosceles_cubic * p**3
    birrell = (
        orders * leading * p**2
        + 0.5 * orders * (orders + 2.0) * birrell_cubic * p**3
    )
    return isosceles, birrell

def scientific_tick(x, pos):
    if x == 0:
        return r"$0$"
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / 10**exponent
    return fr"${coefficient:g}\times 10^{{{exponent}}}$"




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ps", default="0.005,0.01")
    parser.add_argument("--sigma", type=float, default=3)
    parser.add_argument("--alpha-max", type=int, default=15)
    parser.add_argument("--scipy-accuracy", type=float, default=1e-6)
    parser.add_argument("--birrell-taylor-order", type=int, default=4)
    parser.add_argument("--output", default="asymp_renyi_vs_renyi.pdf")
    args = parser.parse_args()

    ps = comma_list(args.ps)
    if min(ps) <= 0.0 or max(ps) >= 1.0 or args.sigma <= 0.0 or args.alpha_max < 2:
        raise ValueError("all p values must lie in (0,1), sigma must be positive, and alpha-max >= 2")
    if args.birrell_taylor_order < 4:
        raise ValueError("the displayed Birrell p^3 expansion requires Taylor order m >= 4")

    orders = np.arange(2, args.alpha_max + 1, dtype=float)
    plt.figure(figsize=(6, 3.5))
    green = np.array([27, 158, 119]) / 255
    brown = np.array([166, 118, 29]) / 255

    sigma = args.sigma
    for index, p in enumerate(ps):
        factor = 0.7 ** (index - 0.5)
        green_color = green * factor
        brown_color = brown * factor
        isosceles_actual = []
        for alpha in orders.astype(int):
            result = estimate_max_renyi(
                p,
                2.0 / sigma,
                alpha,
                accuracy=args.scipy_accuracy,
            )
            if not result.converged:
                raise RuntimeError(f"isosceles calculation did not converge at alpha={alpha}")
            isosceles_actual.append(result.value)
        birrell_actual = fswo_replace_one_rdp_profile(
            orders,
            sigma,
            p,
            taylor_order=args.birrell_taylor_order,
        )
        isosceles_asymptotic, birrell_asymptotic = expansions(
            orders, p, sigma
        )

        plt.plot(orders, isosceles_actual, "o-", color=green_color, markersize=3, label=fr"Our bounds, $p={p:g}$")
        plt.plot(orders, isosceles_asymptotic, "o:", color=green_color, markersize=3, label=fr"Expansion (ours), $p={p:g}$")
        plt.plot(orders, birrell_actual, "s-", color=brown_color, markersize=3, label=fr"Birrell et al., $p={p:g}$")
        plt.plot(orders, birrell_asymptotic, "s-.", color=brown_color, markersize=3, label=fr"Expansion (Birrell et al.), $p={p:g}$")

    plt.xlabel(r"Rényi order $\alpha$")
    plt.xticks(np.arange(5, args.alpha_max+1, 5))
    plt.ylabel("Rényi DP")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_tick))
    plt.title(fr"$\sigma={args.sigma:g}$")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=len(ps) // 2, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
