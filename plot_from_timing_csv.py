"""Create one runtime plot and one TeX table for each (q, sigma, delta)."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


METHODS = [
    ("Our accountant", "isosceles_scipy_binary_search_plus_flint"),
    ("Birrell et al.", "birrell"),
    ("Wang et al.", "wbk"),
]

dark2colors = plt.cm.Dark2.colors
COLORS = [dark2colors[1], dark2colors[2]]

def filename_number(x):
    return f"{x:g}"


def tex_number(value, significant_digits=6):
    coefficient, separator, exponent = f"{value:.{significant_digits}g}".partition("e")
    if not separator:
        return coefficient
    exponent = int(exponent)
    if coefficient == "1":
        return rf"10^{{{exponent}}}"
    return rf"{coefficient}\times 10^{{{exponent}}}"


def quantile(values, probability):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize(values):
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
        quantile(values, 0.1),
        quantile(values, 0.9),
    )


def format_mean_and_standard_deviation(mean, standard_deviation):
    if standard_deviation == 0:
        return tex_number(mean, 4), "0"

    exponent = math.floor(math.log10(abs(standard_deviation)))
    decimal_places = 1 - exponent
    if 0 <= decimal_places <= 8 and abs(mean) < 1e8:
        return (
            f"{mean:.{decimal_places}f}",
            f"{standard_deviation:.{decimal_places}f}",
        )
    return tex_number(mean, 4), tex_number(standard_deviation, 2)


def find_samples_csv(csv_file, explicit_samples_csv):
    if explicit_samples_csv is not None:
        return explicit_samples_csv

    with csv_file.open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or []
    if "elapsed_seconds" in fieldnames:
        return csv_file

    replacement_name = csv_file.name.replace("_timings.csv", "_timing_samples.csv")
    candidates = [csv_file.parent / "dpsgd_timing_samples.csv"]
    if replacement_name != csv_file.name:
        candidates.insert(0, csv_file.with_name(replacement_name))
    candidates.extend(sorted(csv_file.parent.glob("*timing_samples.csv")))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate

    raise ValueError(
        "The 10th and 90th percentiles require the raw timing-sample CSV. "
        "Pass it directly or provide it with --samples-csv."
    )


def write_plot(path, q, sigma, delta, data):
    steps = sorted(data)
    methods = [
        (label, method)
        for label, method in METHODS
        if any(method in data[step] for step in steps)
    ]
    if not methods:
        return

    figure, axis = plt.subplots(figsize=(5, 3.5))
    method_positions = list(range(len(methods)))
    bar_width = 0.8 / max(len(steps), 1)

    for step_index, step in enumerate(steps):
        offset = (step_index - (len(steps) - 1) / 2) * bar_width
        positions = []
        medians = []
        lower_errors = []
        upper_errors = []

        for method_index, (_, method) in enumerate(methods):
            values = data[step].get(method)
            if not values:
                continue
            median = statistics.median(values)
            lower_quantile = quantile(values, 0.1)
            upper_quantile = quantile(values, 0.9)
            positions.append(method_positions[method_index] + offset)
            medians.append(median)
            lower_errors.append(median - lower_quantile)
            upper_errors.append(upper_quantile - median)

        if positions:
            axis.bar(
                positions,
                medians,
                width=bar_width,
                yerr=[lower_errors, upper_errors],
                capsize=4,
                label=rf"$T={step}$",
                color=COLORS[step_index],
                alpha=1 - 0.3*step_index
            )

    axis.set_xticks(
        method_positions,
        labels=[label for label, _ in methods],
    )
    axis.set_ylabel("Time (seconds)")
    axis.set_title(
        rf"$p={q:g},\ \sigma={sigma:g},\ \delta={tex_number(delta)}$"
    )
    axis.grid(axis="y", alpha=0.25)
    if len(steps) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path("results/dpsgd_timing_samples.csv"),
    )
    parser.add_argument("--samples-csv", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    samples_csv = find_samples_csv(args.csv_file, args.samples_csv)
    output_directory = args.output_directory or args.csv_file.parent
    output_directory.mkdir(parents=True, exist_ok=True)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    known_methods = {method for _, method in METHODS}
    with samples_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "sampling_probability_q",
            "noise_multiplier_sigma",
            "steps",
            "privacy_delta",
            "method",
            "elapsed_seconds",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Missing columns in timing-sample CSV: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            if row.get("timing_category") not in (None, "", "comparison_method"):
                continue
            method = row["method"]
            if method not in known_methods:
                continue
            key = (
                float(row["sampling_probability_q"]),
                float(row["noise_multiplier_sigma"]),
                float(row["privacy_delta"]),
            )
            data[key][int(row["steps"])][method].append(float(row["elapsed_seconds"]))

    if not data:
        raise ValueError("No comparison-method timing samples were found.")

    prefix = "dpsgd_timing"
    for (q, sigma, delta), group in sorted(data.items()):
        tag = (
            f"q{filename_number(q)}_"
            f"sigma{filename_number(sigma)}_"
            f"delta{filename_number(delta)}"
        )
        write_plot(output_directory / f"{prefix}_{tag}.pdf", q, sigma, delta, group)

    print(f"Wrote {len(data)} plots to {output_directory}")


if __name__ == "__main__":
    main()
