# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

# Initialize Typer app
app = typer.Typer(
    name="check-metrics-stability",
    help="Analyze stability of benchmark metrics from CSV files.",
    add_completion=False,
)


def analyze_stability(values: list[float]) -> dict[str, float] | None:
    """Analyze the stability of a series of measurements.

    Args:
        values:
            A list of measurements.

    Returns:
        A dictionary with various stability metrics or None when there are
        less than 3 values.
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


def validate_csv_extension(csv_path: Path) -> Path:
    """Validate that the provided path has a .csv extension.

    Args:
        csv_path: Path to the CSV file to validate.

    Returns:
        The validated Path object.

    Raises:
        typer.BadParameter: If the file doesn't have a .csv extension.
    """
    if csv_path.suffix.lower() != ".csv":
        raise typer.BadParameter(
            f"File must have .csv extension, got '{csv_path.suffix}'"
        )
    return csv_path


def perform_stability_analysis(
    df: pd.DataFrame,
    column: str,
) -> None:
    """Perform stability analysis on the specified column of the DataFrame.

    Args:
        df: DataFrame containing the benchmark data.
        column: Name of the column to analyze.
    """
    # Filter rows with valid data
    df = df[df[column].notnull()]

    if df.empty:
        typer.echo(
            f"Error: No valid data found in column '{column}'",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("=" * 80)
    typer.echo(f"STABILITY ANALYSIS: {column}")
    typer.echo("=" * 80)
    typer.echo()

    all_metrics: list[dict[str, float]] = []

    for idx, row in df.iterrows():
        values: Any = row[column]

        if isinstance(values, str):
            try:
                values = json.loads(values)
            except json.JSONDecodeError as e:
                typer.echo(
                    f"Warning: Failed to parse JSON in row {idx}: {e}",
                    err=True,
                )
                continue
        else:
            # Not a string encoded array of numbers. It can contain up to 1
            # data point and that is insufficient
            continue

        metrics = analyze_stability(values)
        if metrics is None:
            continue

        # Store for summary statistics
        all_metrics.append(metrics)

        # Print detailed analysis for this benchmark
        identifier = row.get("identifier", f"Row {idx}")
        typer.echo(f"Benchmark: {identifier}")
        typer.echo(f"  Sample size:        {metrics['count']}")
        typer.echo(f"  Mean:               {metrics['mean']:.2f} tokens/sec/gpu")
        typer.echo(f"  Std Dev:            {metrics['std']:.2f}")
        typer.echo(f"  Min:                {metrics['min']:.2f}")
        typer.echo(f"  Max:                {metrics['max']:.2f}")
        typer.echo(f"  Median:             {metrics['median']:.2f}")
        typer.echo(f"  Range:              {metrics['range']:.2f}")
        typer.echo(f"  CV (Coef. of Var):  {metrics['cv_percent']:.2f}%")
        typer.echo(f"  Range % of Mean:    {metrics['range_percent']:.2f}%")
        typer.echo()

    # Summary statistics across all benchmarks
    if all_metrics:
        _print_summary_statistics(all_metrics)
    else:
        typer.echo(
            "No data to check - make sure you have repetitions of the same "
            "experiment on the same entity and that the data is properly "
            "formatted as JSON arrays.",
            err=True,
        )


def _print_summary_statistics(all_metrics: list[dict[str, float]]) -> None:
    """Print summary statistics across all benchmarks.

    Args:
        all_metrics: List of metric dictionaries from all benchmarks.
    """
    typer.echo("=" * 80)
    typer.echo("SUMMARY STATISTICS ACROSS ALL BENCHMARKS")
    typer.echo("=" * 80)
    typer.echo()

    cv_values: list[float] = [
        m["cv_percent"] for m in all_metrics if np.isfinite(m["cv_percent"])
    ]
    range_pct_values: list[float] = [
        m["range_percent"] for m in all_metrics if np.isfinite(m["range_percent"])
    ]

    typer.echo(f"Total benchmarks analyzed: {len(all_metrics)}")
    typer.echo()
    typer.echo(f'Mean STD {np.mean([x["std"] for x in all_metrics]):.2f}')
    typer.echo(f'STD of STD {np.std([x["std"] for x in all_metrics]):.2f}')
    typer.echo()
    typer.echo("Coefficient of Variation (CV) statistics:")
    typer.echo(f"  Mean CV:    {np.mean(cv_values):.2f}%")
    typer.echo(f"  Median CV:  {np.median(cv_values):.2f}%")
    typer.echo(f"  Min CV:     {np.min(cv_values):.2f}%")
    typer.echo(f"  Max CV:     {np.max(cv_values):.2f}%")
    typer.echo()
    typer.echo("Range as % of Mean statistics:")
    typer.echo(f"  Mean:       {np.mean(range_pct_values):.2f}%")
    typer.echo(f"  Median:     {np.median(range_pct_values):.2f}%")
    typer.echo(f"  Min:        {np.min(range_pct_values):.2f}%")
    typer.echo(f"  Max:        {np.max(range_pct_values):.2f}%")
    typer.echo()

    # Stability assessment
    _print_stability_assessment(cv_values)


def _print_stability_assessment(cv_values: list[float]) -> None:
    """Print overall stability assessment based on CV values.

    Args:
        cv_values: List of coefficient of variation values.
    """
    typer.echo("=" * 80)
    typer.echo("STABILITY ASSESSMENT")
    typer.echo("=" * 80)
    typer.echo()

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

    typer.echo(f"Overall Stability Rating: {stability}")
    typer.echo(f"Description: {desc}")
    typer.echo(f"Median CV: {median_cv:.2f}%")
    typer.echo()
    typer.echo("Interpretation:")
    typer.echo("  - CV < 1%:  Excellent stability")
    typer.echo("  - CV < 2%:  Good stability")
    typer.echo("  - CV < 5%:  Moderate stability")
    typer.echo("  - CV < 10%: Fair stability")
    typer.echo("  - CV ≥ 10%: Poor stability")
    typer.echo()


@app.command()
def main(
    csv_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            callback=validate_csv_extension,
            help="Path to CSV file containing benchmark metrics with JSON-encoded measurement arrays.",
        ),
    ],
    column: Annotated[
        str,
        typer.Option(
            "--column",
            "-c",
            help="Name of the column to analyze for stability metrics. The column should contain JSON arrays of measurements.",
        ),
    ] = "dataset_tokens_per_second_per_gpu",
) -> None:
    """Analyze stability of benchmark metrics from a CSV file.

    This tool reads a CSV file containing benchmark measurements and performs
    comprehensive stability analysis including calculation of coefficient of
    variation (CV), standard deviation, range, and other statistical metrics.

    The CSV file must contain a column with JSON-encoded arrays of repeated
    measurements. Each array represents multiple runs of the same benchmark.

    Example usage:

        \b
        # Analyze default column
        python check_metrics_stability.py benchmarks.csv

        \b
        # Analyze custom column
        python check_metrics_stability.py benchmarks.csv --column throughput_data

        \b
        # Show help
        python check_metrics_stability.py --help
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError as e:
        typer.echo(
            f"Error: CSV file is empty: {csv_file}",
            err=True,
        )
        raise typer.Exit(code=1) from e
    except pd.errors.ParserError as e:
        typer.echo(
            f"Error: Failed to parse CSV file: {e}",
            err=True,
        )
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(
            f"Error: Failed to read CSV file: {e}",
            err=True,
        )
        raise typer.Exit(code=1) from e

    # Validate column exists
    if column not in df.columns:
        typer.echo(
            f"Error: Column '{column}' not found in CSV file.",
            err=True,
        )
        typer.echo(f"Available columns: {', '.join(df.columns)}", err=True)
        raise typer.Exit(code=1)

    # Perform stability analysis
    perform_stability_analysis(df, column)


if __name__ == "__main__":
    app()
