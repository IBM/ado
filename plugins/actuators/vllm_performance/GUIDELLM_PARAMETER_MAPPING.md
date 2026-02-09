# GuideLLM and vLLM Bench Parameter Mapping

This document provides a comprehensive mapping between vLLM bench
parameters and GuideLLM parameters, explaining how the two benchmarking
tools can be used interchangeably within the vLLM performance actuator.

## Overview

The vLLM performance actuator now supports two benchmarking tools:

1. **vLLM bench** - The built-in benchmarking tool from vLLM
2. **GuideLLM** - A comprehensive benchmarking suite for LLM serving
   systems

Both tools measure similar performance metrics but use different
parameter names and have slightly different capabilities.

## GuideLLM CLI Structure

GuideLLM uses a subcommand structure:

```bash
guidellm benchmark run [OPTIONS]
```

## Parameter Mapping Table
<!-- markdownlint-disable line-length -->
| vLLM Bench Parameter | GuideLLM Parameter | Description | Notes |
| --------------------- | ------------------- | ------------- | ------- |
| `--base-url` | `--target` | URL of the LLM endpoint | Both expect OpenAI-compatible endpoints |
| `--model` | `--model` | Model name/identifier | Same parameter name |
| `--num-prompts` | `--max-requests` | Total number of requests to send | GuideLLM uses "max-requests" terminology |
| `--request-rate` | `--profile` + `--rate` | Requests per second | GuideLLM uses profiles (constant, throughput, etc.) |
| `--max-concurrency` | `--max-concurrency` | Maximum concurrent requests | Same parameter name |
| `--random-input-len` | `--data` (JSON) | Number of input tokens per request | Part of synthetic data config: `{"prompt_tokens": N}` |
| `--random-output-len` | `--data` (JSON) | Maximum output tokens per request | Part of synthetic data config: `{"output_tokens": N}` |
| `--backend` | N/A | vLLM backend type | GuideLLM assumes OpenAI-compatible |
| `--burstiness` | N/A | Request distribution pattern | Not directly available in GuideLLM |
| N/A | `--output-path` | Output file path | GuideLLM-specific feature |
| N/A | `--profile` | Benchmark profile type | GuideLLM-specific: constant, throughput, sweep, etc. |
<!-- markdownlint-enable line-length -->

## Key Differences

### 1. Request Rate and Profiles

- **vLLM bench**: Uses `--request-rate` with numeric values or `-1`
  for unlimited
- **GuideLLM**: Uses `--profile` to select benchmarking strategy:
  - `constant`: Fixed request rate (requires `--rate`)
  - `throughput`: Maximum throughput (unlimited rate)
  - `sweep`: Multiple rates (requires `--rate` with multiple values)
  - `poisson`: Poisson-distributed requests
  - `concurrent`: Fixed concurrency level
  - `synchronous`: One request at a time

### 2. Synthetic Data Configuration

- **vLLM bench**: Uses `--random-input-len` and `--random-output-len`
  as separate parameters
- **GuideLLM**: Uses `--data` with JSON config:
  `'{"prompt_tokens": 256, "output_tokens": 128}'`

### 3. Backend Support

- **vLLM bench**: Supports multiple backends (vllm, openai,
  openai-chat, etc.)
- **GuideLLM**: Focuses on OpenAI-compatible endpoints

### 4. Burstiness Control

- **vLLM bench**: Supports `--burstiness` parameter for controlling
  request distribution (Poisson vs Gamma)
- **GuideLLM**: Uses different profiles (poisson, constant) for request
  distribution patterns

### 5. Output Format

- **vLLM bench**: Outputs JSON results to a file
- **GuideLLM**: Supports multiple output formats (JSON, HTML, Markdown)

## Metrics Mapping

Both tools provide similar performance metrics. The implementation
extracts metrics from GuideLLM's JSON output structure and maps them to
the vLLM actuator's expected format.

### GuideLLM JSON Structure

GuideLLM outputs results in the following structure:
<!-- markdownlint-disable line-length -->

```json
{
  "benchmarks": [
    {
      "duration": <seconds>,
      "metrics": {
        "request_totals": {"successful": <count>, ...},
        "prompt_token_count": {"successful": {"total_sum": <tokens>, ...}},
        "output_token_count": {"successful": {"total_sum": <tokens>, ...}},
        "requests_per_second": {"successful": {"mean": <rps>, ...}},
        "output_tokens_per_second": {"successful": {"mean": <tps>, ...}},
        "tokens_per_second": {"successful": {"mean": <tps>, ...}},
        "time_to_first_token_ms": {"successful": {"mean": <ms>, "percentiles": {...}}},
        "time_per_output_token_ms": {"successful": {"mean": <ms>, "percentiles": {...}}},
        "inter_token_latency_ms": {"successful": {"mean": <ms>, "percentiles": {...}}},
        "request_latency": {"successful": {"mean": <seconds>, "percentiles": {...}}}
      }
    }
  ]
}
```
<!-- markdownlint-enable line-length -->

### Metric Mapping Table
<!-- markdownlint-disable line-length -->
| Actuator Output Field | GuideLLM Source Path | Notes |
| ---------------------- | --------------------- | ------- |
| **Basic Metrics** | | |
| `duration` | `benchmarks[0].duration` | Total benchmark duration in seconds |
| `completed` | `benchmarks[0].metrics.request_totals.successful` | Number of successful requests |
| `total_input_tokens` | `benchmarks[0].metrics.prompt_token_count.successful.total_sum` | Total input tokens processed |
| `total_output_tokens` | `benchmarks[0].metrics.output_token_count.successful.total_sum` | Total output tokens generated |
| **Throughput** | | |
| `request_throughput` | `benchmarks[0].metrics.requests_per_second.successful.mean` | Requests per second |
| `output_throughput` | `benchmarks[0].metrics.output_tokens_per_second.successful.mean` | Output tokens per second |
| `total_token_throughput` | `benchmarks[0].metrics.tokens_per_second.successful.mean` | Total tokens per second |
| **TTFT (Time to First Token)** | | |
| `mean_ttft_ms` | `benchmarks[0].metrics.time_to_first_token_ms.successful.mean` | Already in milliseconds |
| `median_ttft_ms` | `benchmarks[0].metrics.time_to_first_token_ms.successful.median` | Already in milliseconds |
| `std_ttft_ms` | `benchmarks[0].metrics.time_to_first_token_ms.successful.std_dev` | Already in milliseconds |
| `p25_ttft_ms` | `benchmarks[0].metrics.time_to_first_token_ms.successful.percentiles.p25` | Already in milliseconds |
| `p50_ttft_ms` | `benchmarks[0].metrics.time_to_first_token_ms.successful.percentiles.p50` | Already in milliseconds |
| `p75_ttft_ms` | `benchmarks[0].metrics.time_to_first_token_ms.successful.percentiles.p75` | Already in milliseconds |
| `p99_ttft_ms` | `benchmarks[0].metrics.time_to_first_token_ms.successful.percentiles.p99` | Already in milliseconds |
| **TPOT (Time Per Output Token)** | | |
| `mean_tpot_ms` | `benchmarks[0].metrics.time_per_output_token_ms.successful.mean` | Already in milliseconds |
| `median_tpot_ms` | `benchmarks[0].metrics.time_per_output_token_ms.successful.median` | Already in milliseconds |
| `std_tpot_ms` | `benchmarks[0].metrics.time_per_output_token_ms.successful.std_dev` | Already in milliseconds |
| `p25_tpot_ms` | `benchmarks[0].metrics.time_per_output_token_ms.successful.percentiles.p25` | Already in milliseconds |
| `p50_tpot_ms` | `benchmarks[0].metrics.time_per_output_token_ms.successful.percentiles.p50` | Already in milliseconds |
| `p75_tpot_ms` | `benchmarks[0].metrics.time_per_output_token_ms.successful.percentiles.p75` | Already in milliseconds |
| `p99_tpot_ms` | `benchmarks[0].metrics.time_per_output_token_ms.successful.percentiles.p99` | Already in milliseconds |
| **ITL (Inter-Token Latency)** | | |
| `mean_itl_ms` | `benchmarks[0].metrics.inter_token_latency_ms.successful.mean` | Already in milliseconds |
| `median_itl_ms` | `benchmarks[0].metrics.inter_token_latency_ms.successful.median` | Already in milliseconds |
| `std_itl_ms` | `benchmarks[0].metrics.inter_token_latency_ms.successful.std_dev` | Already in milliseconds |
| `p25_itl_ms` | `benchmarks[0].metrics.inter_token_latency_ms.successful.percentiles.p25` | Already in milliseconds |
| `p50_itl_ms` | `benchmarks[0].metrics.inter_token_latency_ms.successful.percentiles.p50` | Already in milliseconds |
| `p75_itl_ms` | `benchmarks[0].metrics.inter_token_latency_ms.successful.percentiles.p75` | Already in milliseconds |
| `p99_itl_ms` | `benchmarks[0].metrics.inter_token_latency_ms.successful.percentiles.p99` | Already in milliseconds |
| **E2EL (End-to-End Latency)** | | |
| `mean_e2el_ms` | `benchmarks[0].metrics.request_latency.successful.mean` | Converted from seconds to ms (×1000) |
| `median_e2el_ms` | `benchmarks[0].metrics.request_latency.successful.median` | Converted from seconds to ms (×1000) |
| `std_e2el_ms` | `benchmarks[0].metrics.request_latency.successful.std_dev` | Converted from seconds to ms (×1000) |
| `p25_e2el_ms` | `benchmarks[0].metrics.request_latency.successful.percentiles.p25` | Converted from seconds to ms (×1000) |
| `p50_e2el_ms` | `benchmarks[0].metrics.request_latency.successful.percentiles.p50` | Converted from seconds to ms (×1000) |
| `p75_e2el_ms` | `benchmarks[0].metrics.request_latency.successful.percentiles.p75` | Converted from seconds to ms (×1000) |
| `p99_e2el_ms` | `benchmarks[0].metrics.request_latency.successful.percentiles.p99` | Converted from seconds to ms (×1000) |
<!-- markdownlint-enable line-length -->

### Important Notes

1. **Metric Categories**: Each metric in GuideLLM has subcategories:
   `successful`, `errored`, `incomplete`, and `total`. The
   implementation uses the `successful` category.

2. **Unit Conversions**:
   - TTFT, TPOT, and ITL metrics are already in milliseconds in
     GuideLLM output
   - Request latency (E2EL) is in seconds and must be converted to
     milliseconds (×1000)

3. **Percentiles**: Percentiles are nested under a `percentiles` object
   with keys like `p25`, `p50`, `p75`, `p99`

4. **Statistics**: Each metric category includes: `mean`, `median`,
   `mode`, `variance`, `std_dev`, `min`, `max`, `count`, `total_sum`,
   and `percentiles`
