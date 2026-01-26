# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
Pydantic models for benchmark results

This module defines shared data models for benchmark results that can be used
by both vLLM and GuideLLM benchmarks, ensuring consistent output format.
"""

from pydantic import BaseModel


class BenchmarkResult(BaseModel):
    """
    Standardized benchmark results format

    This model represents the output format used by both vLLM and GuideLLM benchmarks,
    ensuring consistency across different benchmark tools.
    """

    # Basic metrics
    duration: float = 0.0
    completed: int = 0
    total_input_tokens: float = 0.0
    total_output_tokens: float = 0.0

    # Throughput metrics
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    total_token_throughput: float = 0.0

    # Time to First Token (TTFT) metrics - in milliseconds
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    p25_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p75_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0

    # Time Per Output Token (TPOT) metrics - in milliseconds
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    p25_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0
    p75_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    # Inter-Token Latency (ITL) metrics - in milliseconds
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    p25_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p75_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0

    # Request Latency (E2E) metrics - in milliseconds
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    std_e2el_ms: float = 0.0
    p25_e2el_ms: float = 0.0
    p50_e2el_ms: float = 0.0
    p75_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0
