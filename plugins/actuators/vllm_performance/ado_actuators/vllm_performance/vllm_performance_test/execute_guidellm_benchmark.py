# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
GuideLLM Benchmark Execution Module

This module provides functions to execute benchmarks using the GuideLLM benchmark suite
as an alternative to vLLM's built-in benchmarking tools.

GuideLLM is a comprehensive benchmarking tool for LLM serving systems that provides
detailed performance metrics and analysis capabilities.
"""

import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("guidellm-bench")


class GuideLLMBenchmarkError(Exception):
    """Raised if there was an issue when running the GuideLLM benchmark"""


def execute_guidellm_benchmark(
    base_url: str,
    model: str,
    num_prompts: int = 500,
    request_rate: int | None = None,
    max_concurrency: int | None = None,
    benchmark_retries: int = 3,
    retries_timeout: int = 5,
    number_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    dataset: str = "random",
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Execute benchmark using GuideLLM

    GuideLLM Parameter Mapping from vLLM bench:
    - base_url -> --target (OpenAI-compatible endpoint URL)
    - model -> --model (model name/identifier)
    - num_prompts -> --max-requests (total number of requests)
    - request_rate -> --rate (requests per second, use 'inf' for unlimited)
    - max_concurrency -> --max-concurrency (max concurrent requests)
    - number_input_tokens -> --prompt-tokens (input token count)
    - max_output_tokens -> --generation-tokens (output token count)

    :param base_url: URL for the LLM endpoint (OpenAI-compatible)
    :param model: Model name/identifier
    :param num_prompts: Total number of requests to send
    :param request_rate: Request rate (requests per second), None means unlimited
    :param max_concurrency: Maximum number of concurrent requests
    :param benchmark_retries: Number of benchmark execution retries
    :param retries_timeout: Timeout between initial retry
    :param number_input_tokens: Number of input tokens per request
    :param max_output_tokens: Maximum number of output tokens per request
    :param dataset: Dataset type (currently only 'random' is supported for synthetic data)
    :param output_format: Output format (json, html, or markdown)

    :return: Results dictionary with performance metrics

    :raises GuideLLMBenchmarkError if the benchmark failed to execute after
        benchmark_retries attempts
    """

    logger.debug(
        f"Executing GuideLLM benchmark, invoking service at {base_url} with parameters:"
    )
    logger.debug(
        f"model {model}, dataset {dataset}, num_prompts {num_prompts}, request_rate {request_rate}, "
        f"max_concurrency {max_concurrency}"
    )

    # Output to a random file name
    output_dir = Path(".")
    f_name = f"guidellm_{uuid.uuid4().hex}"
    output_path = output_dir / f"{f_name}.{output_format}"

    # Build the guidellm command
    # guidellm benchmark run --target <url> --model <model> [options]
    command = [
        "guidellm",
        "benchmark",
        "run",
        "--target",
        base_url,
        "--model",
        model,
        "--max-requests",
        str(num_prompts),
        "--output-path",
        str(output_path),
    ]

    # Add optional parameters for request rate and profile
    # GuideLLM uses different profiles for different benchmarking strategies
    if request_rate is not None:
        if request_rate <= 0:
            # Negative or zero means unlimited rate - use throughput profile
            command.extend(["--profile", "throughput"])
        else:
            # Use constant profile for fixed request rate
            command.extend(["--profile", "constant"])
            command.extend(["--rate", str(request_rate)])
    else:
        # Default to throughput profile (unlimited rate)
        command.extend(["--profile", "throughput"])

    # Handle synthetic data configuration
    if number_input_tokens is not None or max_output_tokens is not None:
        # Build synthetic data config using JSON format
        # Format: '{"prompt_tokens": 256, "output_tokens": 128}'
        data_config = {}
        if number_input_tokens is not None:
            data_config["prompt_tokens"] = number_input_tokens
        if max_output_tokens is not None:
            data_config["output_tokens"] = max_output_tokens

        command.extend(["--data", json.dumps(data_config)])

    # Log the complete command for debugging
    logger.info(f"Executing GuideLLM command: {' '.join(command)}")

    # Execute the benchmark with retries
    timeout = retries_timeout
    for i in range(benchmark_retries):
        try:
            result = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour timeout for the benchmark
            )
            logger.debug(f"GuideLLM stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"GuideLLM stderr: {result.stderr}")
            break
        except subprocess.CalledProcessError as e:
            logger.warning(f"Command failed with return code {e.returncode}")
            logger.warning(f"stdout: {e.stdout}")
            logger.warning(f"stderr: {e.stderr}")
            if i < benchmark_retries - 1:
                logger.warning(
                    f"Will try again after {timeout} seconds. "
                    f"{benchmark_retries - 1 - i} retries remaining"
                )
                time.sleep(timeout)
                timeout *= 2
            else:
                logger.error(
                    f"Failed to execute GuideLLM benchmark after {benchmark_retries} attempts"
                )
                raise GuideLLMBenchmarkError(
                    f"Failed to execute GuideLLM benchmark: {e.stderr}"
                ) from e
        except subprocess.TimeoutExpired as e:
            logger.error("GuideLLM benchmark timed out after 1 hour")
            raise GuideLLMBenchmarkError(
                "GuideLLM benchmark timed out after 1 hour"
            ) from e

    # Parse the results
    try:
        results = _parse_guidellm_results(output_path)
    except Exception as e:
        logger.error(f"Failed to parse GuideLLM results: {e}")
        raise GuideLLMBenchmarkError(f"Failed to parse GuideLLM results: {e}") from e

    return results


def _parse_guidellm_results(output_path: Path) -> dict[str, Any]:
    """
    Parse GuideLLM benchmark results from output file

    GuideLLM provides comprehensive metrics including:
    - Request throughput (requests/sec)
    - Token throughput (tokens/sec)
    - Time to First Token (TTFT) statistics
    - Time Per Output Token (TPOT) statistics
    - Inter-Token Latency (ITL) statistics
    - Request Latency statistics
    - Success/failure rates

    This function maps GuideLLM metrics to the format expected by the vLLM actuator.

    :param output_path: Path to the GuideLLM output file
    :return: Dictionary with parsed metrics
    """

    if not output_path.exists():
        raise FileNotFoundError(f"GuideLLM output file not found: {output_path}")

    with open(output_path) as f:
        if output_path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

    # Extract the first benchmark result (typically there's only one)
    if "benchmarks" not in data or len(data["benchmarks"]) == 0:
        raise ValueError("No benchmark results found in GuideLLM output")

    benchmark = data["benchmarks"][0]
    metrics = benchmark.get("metrics", {})

    # Helper function to safely extract metric values
    def get_metric(metric_name: str, stat: str, category: str = "successful") -> float:
        """Extract a specific statistic from a metric category"""
        try:
            value = metrics.get(metric_name, {}).get(category, {}).get(stat, 0)
            return float(value) if value is not None else 0.0
        except (KeyError, TypeError, ValueError):
            return 0.0

    def get_percentile(
        metric_name: str, percentile: str, category: str = "successful"
    ) -> float:
        """Extract a specific percentile from a metric category"""
        try:
            percentiles = (
                metrics.get(metric_name, {}).get(category, {}).get("percentiles", {})
            )
            if isinstance(percentiles, dict):
                return float(percentiles.get(percentile, 0))
            return 0.0
        except (KeyError, TypeError, ValueError):
            return 0.0

    # Calculate duration from benchmark timing
    duration = benchmark.get("duration", 0)

    # Extract request counts
    request_totals = metrics.get("request_totals", {})
    completed = (
        request_totals.get("successful", {}).get("count", 0)
        if isinstance(request_totals.get("successful"), dict)
        else request_totals.get("successful", 0)
    )

    # Extract token counts
    prompt_tokens_total = get_metric("prompt_token_count", "total_sum", "successful")
    output_tokens_total = get_metric("output_token_count", "total_sum", "successful")

    # Map GuideLLM metrics to vLLM bench format
    # This mapping ensures compatibility with existing result processing
    return {
        # Basic metrics
        "duration": duration,
        "completed": completed,
        "total_input_tokens": prompt_tokens_total,
        "total_output_tokens": output_tokens_total,
        # Throughput metrics
        "request_throughput": get_metric("requests_per_second", "mean", "successful"),
        "output_throughput": get_metric(
            "output_tokens_per_second", "mean", "successful"
        ),
        "total_token_throughput": get_metric("tokens_per_second", "mean", "successful"),
        # Time to First Token (TTFT) metrics - in milliseconds
        "mean_ttft_ms": get_metric("time_to_first_token_ms", "mean", "successful"),
        "median_ttft_ms": get_metric("time_to_first_token_ms", "median", "successful"),
        "std_ttft_ms": get_metric("time_to_first_token_ms", "std_dev", "successful"),
        "p25_ttft_ms": get_percentile("time_to_first_token_ms", "p25", "successful"),
        "p50_ttft_ms": get_percentile("time_to_first_token_ms", "p50", "successful"),
        "p75_ttft_ms": get_percentile("time_to_first_token_ms", "p75", "successful"),
        "p99_ttft_ms": get_percentile("time_to_first_token_ms", "p99", "successful"),
        # Time Per Output Token (TPOT) metrics - in milliseconds
        "mean_tpot_ms": get_metric("time_per_output_token_ms", "mean", "successful"),
        "median_tpot_ms": get_metric(
            "time_per_output_token_ms", "median", "successful"
        ),
        "std_tpot_ms": get_metric("time_per_output_token_ms", "std_dev", "successful"),
        "p25_tpot_ms": get_percentile("time_per_output_token_ms", "p25", "successful"),
        "p50_tpot_ms": get_percentile("time_per_output_token_ms", "p50", "successful"),
        "p75_tpot_ms": get_percentile("time_per_output_token_ms", "p75", "successful"),
        "p99_tpot_ms": get_percentile("time_per_output_token_ms", "p99", "successful"),
        # Inter-Token Latency (ITL) metrics - in milliseconds
        "mean_itl_ms": get_metric("inter_token_latency_ms", "mean", "successful"),
        "median_itl_ms": get_metric("inter_token_latency_ms", "median", "successful"),
        "std_itl_ms": get_metric("inter_token_latency_ms", "std_dev", "successful"),
        "p25_itl_ms": get_percentile("inter_token_latency_ms", "p25", "successful"),
        "p50_itl_ms": get_percentile("inter_token_latency_ms", "p50", "successful"),
        "p75_itl_ms": get_percentile("inter_token_latency_ms", "p75", "successful"),
        "p99_itl_ms": get_percentile("inter_token_latency_ms", "p99", "successful"),
        # Request Latency (E2E) metrics - convert from seconds to milliseconds
        "mean_e2el_ms": get_metric("request_latency", "mean", "successful") * 1000,
        "median_e2el_ms": get_metric("request_latency", "median", "successful") * 1000,
        "std_e2el_ms": get_metric("request_latency", "std_dev", "successful") * 1000,
        "p25_e2el_ms": get_percentile("request_latency", "p25", "successful") * 1000,
        "p50_e2el_ms": get_percentile("request_latency", "p50", "successful") * 1000,
        "p75_e2el_ms": get_percentile("request_latency", "p75", "successful") * 1000,
        "p99_e2el_ms": get_percentile("request_latency", "p99", "successful") * 1000,
    }


if __name__ == "__main__":
    # Example usage
    results = execute_guidellm_benchmark(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-3.1-8B-Instruct",
        num_prompts=100,
        request_rate=10,
        max_concurrency=20,
        number_input_tokens=1024,
        max_output_tokens=128,
    )
    print(json.dumps(results, indent=2))

# Made with Bob
