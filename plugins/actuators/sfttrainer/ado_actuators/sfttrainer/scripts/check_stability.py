# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import sys

import numpy as np
import pandas as pd


def analyze_stability(values: list[float]) -> dict[str, float] | None:
    """Analyze the stability of a series of measurements.

    Args:
        values:
            A list of measurements.

    Returns:
        a dictionary with various stability metrics or None when there are less than 3 values
    """
    n = len(values)

    if n < 3:
        return None

    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0

    metrics = {
        "count": n,
        "mean": mean,
        "std": std,
        "min": np.min(values),
        "max": np.max(values),
        "range": np.max(values) - np.min(values),
        "median": np.median(values),
    }

    # Coefficient of Variation (CV) - relative variability
    if mean != 0:
        metrics["cv_percent"] = (std / mean) * 100
    else:
        metrics["cv_percent"] = float("inf")

    # Range as percentage of mean
    if mean != 0:
        metrics["range_percent"] = (metrics["range"] / mean) * 100
    else:
        metrics["range_percent"] = float("inf")

    return metrics


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python check.py <csv_file>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    column = "dataset_tokens_per_second_per_gpu"

    # Filter rows with valid data
    df = df[df[column].notnull()]

    print("=" * 80)
    print("STABILITY ANALYSIS: ", column)
    print("=" * 80)
    print()

    all_metrics: list[dict[str, float]] = []

    for idx, row in df.iterrows():
        values = row[column]

        if isinstance(values, str):
            values = json.loads(row[column])
        else:
            # VV: Not an string encoded array of numbers. It can contain up to 1
            # data point and that is insufficient
            continue

        metrics = analyze_stability(values)
        if metrics is None:
            continue

        # Store for summary statistics
        all_metrics.append(metrics)

        # Print detailed analysis for this benchmark
        identifier = row.get("identifier", f"Row {idx}")
        print(f"Benchmark: {identifier}")
        print(f"  Sample size:        {metrics['count']}")
        print(f"  Mean:               {metrics['mean']:.2f} tokens/sec/gpu")
        print(f"  Std Dev:            {metrics['std']:.2f}")
        print(f"  Min:                {metrics['min']:.2f}")
        print(f"  Max:                {metrics['max']:.2f}")
        print(f"  Median:             {metrics['median']:.2f}")
        print(f"  Range:              {metrics['range']:.2f}")
        print(f"  CV (Coef. of Var):  {metrics['cv_percent']:.2f}%")
        print(f"  Range % of Mean:    {metrics['range_percent']:.2f}%")
        print()

    # Summary statistics across all benchmarks
    if all_metrics:
        print("=" * 80)
        print("SUMMARY STATISTICS ACROSS ALL BENCHMARKS")
        print("=" * 80)
        print()

        cv_values: list[float] = [
            m["cv_percent"] for m in all_metrics if np.isfinite(m["cv_percent"])
        ]
        range_pct_values: list[float] = [
            m["range_percent"] for m in all_metrics if np.isfinite(m["range_percent"])
        ]

        print(f"Total benchmarks analyzed: {len(all_metrics)}")
        print()
        print(f'Mean STD {np.mean([x["std"] for x in all_metrics]):.2f}')
        print(f'STD of STD {np.std([x["std"] for x in all_metrics]):.2f}')
        print()
        print("Coefficient of Variation (CV) statistics:")
        print(f"  Mean CV:    {np.mean(cv_values):.2f}%")
        print(f"  Median CV:  {np.median(cv_values):.2f}%")
        print(f"  Min CV:     {np.min(cv_values):.2f}%")
        print(f"  Max CV:     {np.max(cv_values):.2f}%")
        print()
        print("Range as % of Mean statistics:")
        print(f"  Mean:       {np.mean(range_pct_values):.2f}%")
        print(f"  Median:     {np.median(range_pct_values):.2f}%")
        print(f"  Min:        {np.min(range_pct_values):.2f}%")
        print(f"  Max:        {np.max(range_pct_values):.2f}%")
        print()

        # Stability assessment
        print("=" * 80)
        print("STABILITY ASSESSMENT")
        print("=" * 80)
        print()

        median_cv = np.median(cv_values)

        if median_cv < 1.0:
            stability = "EXCELLENT"
            desc = "Very stable measurements with minimal variation"
        elif median_cv < 2.0:
            stability = "GOOD"
            desc = "Stable measurements with low variation"
        elif median_cv < 5.0:
            stability = "MODERATE"
            desc = "Acceptable stability with some variation"
        elif median_cv < 10.0:
            stability = "FAIR"
            desc = "Noticeable variation in measurements"
        else:
            stability = "POOR"
            desc = "High variation in measurements"

        print(f"Overall Stability Rating: {stability}")
        print(f"Description: {desc}")
        print(f"Median CV: {median_cv:.2f}%")
        print()
        print("Interpretation:")
        print("  - CV < 1%:  Excellent stability")
        print("  - CV < 2%:  Good stability")
        print("  - CV < 5%:  Moderate stability")
        print("  - CV < 10%: Fair stability")
        print("  - CV ≥ 10%: Poor stability")
        print()
    else:
        print(
            "No data to check - make sure you have repetitions of the same experiment on the same entity and that"
        )


if __name__ == "__main__":
    main()
