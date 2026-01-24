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
from typing import Any, Literal

logger = logging.getLogger("guidellm-bench")


class GuideLLMBenchmarkError(Exception):
    """Raised if there was an issue when running the GuideLLM benchmark"""


def execute_guidellm_benchmark(
    base_url: str,
    model: str,
    num_prompts: int = 500,
    request_rate: int | None = None,
    max_concurrency: int | None = None,
    hf_token: str | None = None,
    benchmark_retries: int = 3,
    retries_timeout: int = 5,
    number_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    dataset: Literal["random"] = "random",
    output_format: Literal["json", "html", "markdown"] = "json",
    burstiness: float = 1.0,
) -> dict[str, Any]:
    """
    Execute benchmark using GuideLLM

    GuideLLM Parameter Mapping from vLLM bench:
    - base_url -> --target (OpenAI-compatible endpoint URL)
    - model -> --model (model name/identifier)
    - num_prompts -> --max-requests (total number of requests)
    - request_rate -> --rate (requests per second) with --profile poisson
    - max_concurrency -> --max-concurrency (max concurrent requests)
    - number_input_tokens -> --prompt-tokens (input token count)
    - max_output_tokens -> --generation-tokens (output token count)
    - burstiness -> Must be 1.0 (GuideLLM uses poisson profile for Poisson-distributed requests)

    :param base_url: URL for the LLM endpoint (OpenAI-compatible)
    :param model: Model name/identifier
    :param num_prompts: Total number of requests to send (must be positive)
    :param request_rate: Request rate (requests per second), None means unlimited
    :param max_concurrency: Maximum number of concurrent requests (must be positive if specified)
    :param hf_token: HuggingFace token for authentication
    :param benchmark_retries: Number of benchmark execution retries (must be positive)
    :param retries_timeout: Timeout between initial retry (must be positive)
    :param number_input_tokens: Number of input tokens per request (must be positive if specified)
    :param max_output_tokens: Maximum number of output tokens per request (must be positive if specified)
    :param dataset: Dataset type (currently only 'random' is supported for synthetic data)
    :param output_format: Output format (json, html, or markdown)
    :param burstiness: Burstiness parameter (must be 1.0 for GuideLLM experiments, default: 1.0)

    :return: Results dictionary with performance metrics

    :raises ValueError: If any numeric parameter is invalid or burstiness != 1.0
    :raises GuideLLMBenchmarkError if the benchmark failed to execute after
        benchmark_retries attempts
    """

    # GuideLLM does not support the burstiness parameter. For GuideLLM we only support `burstiness = 1`
    if burstiness != 1.0:
        raise ValueError(
            f"GuideLLM experiments only support burstiness=1.0 (Poisson distribution), got {burstiness}"
        )

    # Validate positive integer parameters
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}")
    if benchmark_retries <= 0:
        raise ValueError(f"benchmark_retries must be positive, got {benchmark_retries}")
    if retries_timeout <= 0:
        raise ValueError(f"retries_timeout must be positive, got {retries_timeout}")
    if number_input_tokens is not None and number_input_tokens <= 0:
        raise ValueError(
            f"number_input_tokens must be positive, got {number_input_tokens}"
        )
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError(f"max_output_tokens must be positive, got {max_output_tokens}")
    if max_concurrency is not None and max_concurrency <= 0:
        raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
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
    if request_rate is None or request_rate <= 0:
        # None, negative or zero means unlimited rate - use throughput profile
        command.extend(["--profile", "throughput"])
    else:
        # For completentess with respect to the vLLM bench
        # case where `burtiness = 1`` selects a poisson distribution, we force `burstiness = 1`
        # for GuideLLM and default on the poisson profile.
        command.extend(["--profile", "poisson"])
        command.extend(["--rate", str(request_rate)])

    # Add max_concurrency if specified (already validated above)
    if max_concurrency is not None:
        command.extend(["--max-concurrency", str(max_concurrency)])

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

    # Set up environment variables
    import os

    env = os.environ.copy()
    if hf_token is not None:
        env["HF_TOKEN"] = hf_token

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
                env=env,
            )
            logger.debug(f"GuideLLM stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"GuideLLM stderr: {result.stderr}")
            break
        except subprocess.CalledProcessError as e:
            logger.warning(f"Command failed with return code {e.returncode}")
            logger.warning(f"stdout: {e.stdout}")
            logger.warning(f"stderr: {e.stderr}")

            if i == benchmark_retries - 1:
                logger.error(
                    f"Failed to execute GuideLLM benchmark after {benchmark_retries} attempts"
                )
                raise GuideLLMBenchmarkError(
                    f"Failed to execute GuideLLM benchmark: {e.stderr}"
                ) from e

            logger.warning(
                f"Will try again after {timeout} seconds. "
                f"{benchmark_retries - 1 - i} retries remaining"
            )
            time.sleep(timeout)
            timeout *= 2

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

    if output_path.suffix != ".json":
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

    data = json.loads(output_path.read_text())

    # Extract the first benchmark result (typically there's only one)
    if "benchmarks" not in data or len(data["benchmarks"]) == 0:
        raise ValueError("No benchmark results found in GuideLLM output")

    benchmark = data["benchmarks"][0]
    metrics = benchmark.get("metrics", {})

    # Helper function to safely extract metric values
    def get_metric(
        metric_name: str,
        stat: str,
        category: Literal["successful", "failed"] = "successful",
    ) -> float:
        """Extract a specific statistic from a metric category"""
        try:
            value = metrics.get(metric_name, {}).get(category, {}).get(stat, 0.0)
            return float(value) if value is not None else 0.0
        except TypeError:
            # TypeError occurs when value cannot be converted to float
            # This indicates the metric format may have changed
            logger.warning(
                f"Failed to convert metric {metric_name}.{category}.{stat} to float"
            )
            return 0.0

    def get_percentile(
        metric_name: str,
        percentile: str,
        category: Literal["successful", "failed"] = "successful",
    ) -> float:
        """Extract a specific percentile from a metric category"""
        try:
            percentiles = (
                metrics.get(metric_name, {}).get(category, {}).get("percentiles", {})
            )
            if isinstance(percentiles, dict):
                return float(percentiles.get(percentile, 0.0))
            return 0.0
        except TypeError:
            # TypeError occurs when value cannot be converted to float
            # This indicates the metric format may have changed
            logger.warning(
                f"Failed to convert percentile {metric_name}.{category}.{percentile} to float"
            )
            return 0.0

    # Extract duration from benchmark timing
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
