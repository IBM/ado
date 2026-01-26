# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
Pydantic models for GuideLLM benchmark output structure

This module defines the data models for parsing and validating GuideLLM
benchmark JSON output and transforming it into the standardized BenchmarkResult format.
"""

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkResult


class MetricPercentiles(BaseModel):
    """Percentile statistics for a metric"""

    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p99: float = 0.0


class MetricCategory(BaseModel):
    """Statistics for a metric category (successful/failed)"""

    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    total_sum: float = 0.0
    percentiles: MetricPercentiles = Field(default_factory=MetricPercentiles)


class Metric(BaseModel):
    """A metric with successful and failed categories"""

    successful: MetricCategory = Field(default_factory=MetricCategory)
    failed: MetricCategory = Field(default_factory=MetricCategory)


class RequestTotals(BaseModel):
    """Request totals with successful and failed counts"""

    successful: int | MetricCategory = 0
    failed: int = 0


class BenchmarkMetrics(BaseModel):
    """All metrics from a GuideLLM benchmark"""

    request_totals: RequestTotals = Field(default_factory=RequestTotals)
    prompt_token_count: Metric = Field(default_factory=Metric)
    output_token_count: Metric = Field(default_factory=Metric)
    requests_per_second: Metric = Field(default_factory=Metric)
    output_tokens_per_second: Metric = Field(default_factory=Metric)
    tokens_per_second: Metric = Field(default_factory=Metric)
    time_to_first_token_ms: Metric = Field(default_factory=Metric)
    time_per_output_token_ms: Metric = Field(default_factory=Metric)
    inter_token_latency_ms: Metric = Field(default_factory=Metric)
    request_latency: Metric = Field(default_factory=Metric)


class Benchmark(BaseModel):
    """A single benchmark result from GuideLLM"""

    duration: float = 0.0
    metrics: BenchmarkMetrics = Field(default_factory=BenchmarkMetrics)


class GuideLLMOutput(BaseModel):
    """Root structure of GuideLLM JSON output"""

    benchmarks: list[Benchmark] = Field(default_factory=list)

    def to_benchmark_result(self) -> BenchmarkResult:
        """
        Transform GuideLLM output to standardized BenchmarkResult format

        :return: BenchmarkResult with mapped metrics
        :raises ValueError: If no benchmarks are found in the output
        """
        if not self.benchmarks:
            raise ValueError("No benchmark results found in GuideLLM output")

        benchmark = self.benchmarks[0]
        metrics = benchmark.metrics

        # Extract request counts - handle both int and MetricCategory types
        completed = (
            metrics.request_totals.successful.count
            if isinstance(metrics.request_totals.successful, MetricCategory)
            else metrics.request_totals.successful
        )

        return BenchmarkResult(
            # Basic metrics
            duration=benchmark.duration,
            completed=completed,
            total_input_tokens=metrics.prompt_token_count.successful.total_sum,
            total_output_tokens=metrics.output_token_count.successful.total_sum,
            # Throughput metrics
            request_throughput=metrics.requests_per_second.successful.mean,
            output_throughput=metrics.output_tokens_per_second.successful.mean,
            total_token_throughput=metrics.tokens_per_second.successful.mean,
            # Time to First Token (TTFT) metrics - in milliseconds
            mean_ttft_ms=metrics.time_to_first_token_ms.successful.mean,
            median_ttft_ms=metrics.time_to_first_token_ms.successful.median,
            std_ttft_ms=metrics.time_to_first_token_ms.successful.std_dev,
            p25_ttft_ms=metrics.time_to_first_token_ms.successful.percentiles.p25,
            p50_ttft_ms=metrics.time_to_first_token_ms.successful.percentiles.p50,
            p75_ttft_ms=metrics.time_to_first_token_ms.successful.percentiles.p75,
            p99_ttft_ms=metrics.time_to_first_token_ms.successful.percentiles.p99,
            # Time Per Output Token (TPOT) metrics - in milliseconds
            mean_tpot_ms=metrics.time_per_output_token_ms.successful.mean,
            median_tpot_ms=metrics.time_per_output_token_ms.successful.median,
            std_tpot_ms=metrics.time_per_output_token_ms.successful.std_dev,
            p25_tpot_ms=metrics.time_per_output_token_ms.successful.percentiles.p25,
            p50_tpot_ms=metrics.time_per_output_token_ms.successful.percentiles.p50,
            p75_tpot_ms=metrics.time_per_output_token_ms.successful.percentiles.p75,
            p99_tpot_ms=metrics.time_per_output_token_ms.successful.percentiles.p99,
            # Inter-Token Latency (ITL) metrics - in milliseconds
            mean_itl_ms=metrics.inter_token_latency_ms.successful.mean,
            median_itl_ms=metrics.inter_token_latency_ms.successful.median,
            std_itl_ms=metrics.inter_token_latency_ms.successful.std_dev,
            p25_itl_ms=metrics.inter_token_latency_ms.successful.percentiles.p25,
            p50_itl_ms=metrics.inter_token_latency_ms.successful.percentiles.p50,
            p75_itl_ms=metrics.inter_token_latency_ms.successful.percentiles.p75,
            p99_itl_ms=metrics.inter_token_latency_ms.successful.percentiles.p99,
            # Request Latency (E2E) metrics - convert from seconds to milliseconds
            mean_e2el_ms=metrics.request_latency.successful.mean * 1000,
            median_e2el_ms=metrics.request_latency.successful.median * 1000,
            std_e2el_ms=metrics.request_latency.successful.std_dev * 1000,
            p25_e2el_ms=metrics.request_latency.successful.percentiles.p25 * 1000,
            p50_e2el_ms=metrics.request_latency.successful.percentiles.p50 * 1000,
            p75_e2el_ms=metrics.request_latency.successful.percentiles.p75 * 1000,
            p99_e2el_ms=metrics.request_latency.successful.percentiles.p99 * 1000,
        )
