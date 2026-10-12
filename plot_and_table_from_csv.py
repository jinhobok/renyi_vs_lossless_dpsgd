"""Read a comparison CSV and write one TeX table and PDF plot per parameter pair."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

METHODS = [
    ("Our bounds", ("isosceles_epsilon_upper",)),
    ("Birrell et al.", ("birrell_epsilon",)),
    ("Wang et al.", ("wbk_epsilon",)),
    (
        "Lossless accountants",
        ("connect_the_dots_epsilon_upper",),
    ),
]
dark2colors = plt.cm.Dark2.colors
STYLES = ["-", "--", "-.", ":"]
COLORS = [dark2colors[0], dark2colors[6], dark2colors[1], dark2colors[2]]
MARKERS = ["o", "s", "D", "^"]


def tex_number(x):
    coefficient, separator, exponent = f"{x:g}".partition("e")
    if not separator:
        return coefficient
    exponent = int(exponent)
    return rf"10^{{{exponent}}}" if coefficient == "1" else rf"{coefficient}\times10^{{{exponent}}}"


def filename_number(x):
    return f"{x:g}"


def _optional_number(row, column):
    """Return a finite numeric CSV value, or None for a recorded error."""
    raw = row.get(column, "")
    text = str(raw).strip()
    if not text or text.lower() == "error":
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid value {raw!r} in CSV column {column!r}."
        ) from exc
    return value if math.isfinite(value) else None


def method_value(row, columns):
    values = [
        value
        for column in columns
        if (value := _optional_number(row, column)) is not None
    ]
    if not values:
        parameters = ", ".join(
            f"{name}={row.get(name, '?')}"
            for name in (
                "sampling_probability_q",
                "noise_multiplier_sigma",
                "steps",
                "privacy_delta",
            )
        )
        raise ValueError(
            "No successful value is available among "
            f"{tuple(columns)} for {parameters}."
        )
    return min(values)


def improvement_percentage(row):
    """Return 100 * (our bound - lossless bound) / lossless bound."""
    our_bound = method_value(row, METHODS[0][1])
    lossless_bound = method_value(row, METHODS[-1][1])
    if lossless_bound == 0.0:
        raise ValueError("Cannot compute Improvement when the lossless bound is zero.")
    return 100.0 * (our_bound - lossless_bound) / lossless_bound


def tex_improvement(value):
    """Format an improvement percentage, colored by its unrounded sign."""
    color = "Green" if value < 0.0 else "Red"
    rounded = round(value, 2)
    sign = "-" if value < 0.0 else "+"
    return rf"$\textcolor{{{color}}}{{{sign}{abs(rounded):.2f}\%}}$"


def tex_epsilon(value):
    if value >= 100.0:
        return r"$\geq 100$"
    return rf"${round(value, 2):.2f}$"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path("results/dpsgd_comparison.csv"),
    )
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    output_directory = args.output_directory or args.csv_file.parent
    output_directory.mkdir(parents=True, exist_ok=True)

    with args.csv_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    groups = defaultdict(list)
    for row in rows:
        key = tuple(
            float(row[name])
            for name in (
                "sampling_probability_q",
                "noise_multiplier_sigma",
                "privacy_delta",
            )
        )
        groups[key].append(row)

    prefix = args.csv_file.stem
    tex_path = output_directory / f"{prefix}_tables.tex"

    for (q, sigma, delta), group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["steps"]))
        T = [int(row["steps"]) for row in group]
        if len(groups) == 1:
            plot_name = f"{prefix}.pdf"
        else:
            plot_name = (
                f"{prefix}_q{filename_number(q)}"
                f"_sigma{filename_number(sigma)}"
                f"_delta{filename_number(delta)}.pdf"
            )

        figure, axis = plt.subplots(figsize=(5, 3))
        for (label, columns), style, color, marker in zip(METHODS, STYLES, COLORS, MARKERS):
            axis.plot(
                T,
                [method_value(row, columns) for row in group],
                marker=marker,
                color=color,
                linestyle=style,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_xticks(T, labels=[str(value) for value in T])
        axis.set_xlabel(r"Iterations $T$")
        axis.set_ylabel(r"Privacy $\varepsilon$")
        axis.set_title(rf"$p={q:g},\ \sigma={sigma:g},\ \delta={tex_number(delta)}$")
        axis.grid(alpha=0.3)
        axis.legend(ncol=1, loc="upper left")
        figure.savefig(
            output_directory / plot_name,
            bbox_inches="tight",
        )
        plt.close(figure)

    table_groups = defaultdict(list)
    for row in rows:
        table_groups[(float(row["sampling_probability_q"]), float(row["privacy_delta"]))].append(row)

    with tex_path.open("w", encoding="utf-8") as tex:
        print(r"% Requires \usepackage{multirow} and \usepackage[dvipsnames]{xcolor}.", file=tex)
        for (q, delta), table_rows in sorted(table_groups.items()):
            sigma_groups = defaultdict(list)
            for row in table_rows:
                sigma_groups[float(row["noise_multiplier_sigma"])].append(row)
            for sigma_rows in sigma_groups.values():
                sigma_rows.sort(key=lambda row: int(row["steps"]))

            print(r"\begin{table}[ht]", file=tex)
            print(r"\centering", file=tex)
            print(
                rf"\caption{{Accounted privacy $\varepsilon$ for $\delta={tex_number(delta)}$ and $p={q:g}$.}}",
                file=tex,
            )
            print(r"\begin{tabular}{|c|c|" + "c|" * (len(METHODS) + 1) + "}", file=tex)
            print(r"\hline", file=tex)
            print(
                r"\multirow{2}{*}{$\sigma$} & \multirow{2}{*}{$T$} & "
                r"\multicolumn{3}{c|}{R\'enyi accountants} & "
                r"\multirow{2}{*}{Lossless accountants} & "
                r"\multirow{2}{*}{Relative difference} " + r"\\",
                file=tex,
            )
            print(r"\cline{3-5}", file=tex)
            print(
                r" &  & Our bounds & Birrell et al. & Wang et al. &  &  " + r"\\",
                file=tex,
            )
            print(r"\hline", file=tex)
            sorted_sigmas = sorted(sigma_groups)
            for sigma_index, sigma in enumerate(sorted_sigmas):
                sigma_rows = sigma_groups[sigma]
                cutoff_index = next(
                    (
                        row_index
                        for row_index, row in enumerate(sigma_rows)
                        if all(
                            method_value(row, columns) >= 20.0
                            for _, columns in METHODS
                        )
                    ),
                    None,
                )
                displayed_rows = (
                    sigma_rows
                    if cutoff_index is None
                    else sigma_rows[:cutoff_index + 1]
                )
                for row_index, row in enumerate(displayed_rows):
                    sigma_entry = (
                        rf"\multirow{{{len(displayed_rows)}}}{{*}}{{${sigma:g}$}}"
                        if row_index == 0
                        else ""
                    )
                    if cutoff_index is not None and row_index == cutoff_index:
                        print(r"\cline{3-7}", file=tex)
                        print(
                            " & ".join(
                                [
                                    sigma_entry,
                                    rf"$\geq {row['steps']}$",
                                    r"\multicolumn{5}{c|}{$\geq 20$ for all accountants}",
                                ]
                            )
                            + r" \\",
                            file=tex,
                        )
                    else:
                        values = [
                            sigma_entry,
                            rf"${row['steps']}$",
                        ] + [
                            tex_epsilon(method_value(row, columns))
                            for _, columns in METHODS
                        ] + [
                            tex_improvement(improvement_percentage(row))
                        ]
                        print(" & ".join(values) + r" \\", file=tex)
                if sigma_index + 1 < len(sorted_sigmas):
                    print(r"\hline", file=tex)
            print(r"\hline", file=tex)
            print(r"\end{tabular}", file=tex)
            print(r"\end{table}", file=tex)
            print(file=tex)

    print(f"Wrote {tex_path}")
    print(f"Wrote {len(groups)} plots to {output_directory}")


if __name__ == "__main__":
    main()
