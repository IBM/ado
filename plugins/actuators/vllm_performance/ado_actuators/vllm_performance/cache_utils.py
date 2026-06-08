# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utilities for building cache keys for vLLM performance measurements."""

import json

from ado_actuators.vllm_performance.version_utils import VLLMVersionChecker


class CacheKeyBuilder:
    """Build cache keys for vLLM performance measurements.

    Cache keys combine environment parameters (model, GPUs, etc.) and
    benchmark parameters (num_prompts, request_rate, etc.) to ensure
    measurements are only reused for identical configurations.
    """

    # Environment parameters that define the deployment
    ENV_PARAMS = [
        "model",
        "image",
        "n_gpus",
        "gpu_type",
        "n_cpus",
        "memory",
        "max_batch_tokens",
        "gpu_memory_utilization",
        "dtype",
        "cpu_offload",
        "max_num_seq",
        "threadpool",
        "renderer_num_workers",
    ]

    BENCHMARK_PARAMS: list[str] = [
        "num_prompts",
        "request_rate",
        "max_concurrency",
        "number_input_tokens",
        "max_output_tokens",
        "burstiness",
        "dataset",
    ]

    # All parameters used in cache key
    ALL_PARAMS = ENV_PARAMS + BENCHMARK_PARAMS

    @classmethod
    def _normalize_and_extract_env_params(cls, values: dict[str, str]) -> dict:
        """Extract and normalize environment parameters."""
        # Extract and normalize image value
        image_value = values.get("image")
        if isinstance(image_value, list):
            image_str = image_value[0] if len(image_value) > 0 else image_value
        else:
            image_str = image_value

        # Normalize threadpool properties based on vLLM version
        threadpool_requested = int(values.get("threadpool", 1))
        renderer_num_workers_requested = int(values.get("renderer_num_workers", 32))

        enable_threadpool = VLLMVersionChecker.supports_threadpool(
            image_value if image_value is not None else "", threadpool_requested
        )

        if enable_threadpool:
            threadpool_value = 1
            renderer_num_workers_value = renderer_num_workers_requested
        else:
            threadpool_value = 0
            renderer_num_workers_value = 0

        return {
            "model": values.get("model"),
            "image": image_str,
            "n_gpus": values.get("n_gpus"),
            "gpu_type": values.get("gpu_type"),
            "n_cpus": values.get("n_cpus"),
            "memory": values.get("memory"),
            "max_batch_tokens": values.get("max_batch_tokens"),
            "gpu_memory_utilization": values.get("gpu_memory_utilization"),
            "dtype": values.get("dtype"),
            "cpu_offload": values.get("cpu_offload"),
            "max_num_seq": values.get("max_num_seq"),
            "threadpool": threadpool_value,
            "renderer_num_workers": renderer_num_workers_value,
        }

    @classmethod
    def build_env_definition(cls, values: dict[str, str]) -> str:
        """Build environment definition JSON string."""
        env_values = cls._normalize_and_extract_env_params(values)
        return json.dumps(env_values)

    @classmethod
    def build(cls, values: dict[str, str]) -> str:
        """Build composite cache key from environment and benchmark parameters."""
        env_values = cls._normalize_and_extract_env_params(values)

        # Build benchmark parameters
        benchmark_params = {
            "num_prompts": values.get("num_prompts"),
            "request_rate": values.get("request_rate"),
            "max_concurrency": values.get("max_concurrency"),
            "number_input_tokens": values.get("number_input_tokens"),
            "max_output_tokens": values.get("max_output_tokens"),
            "burstiness": values.get("burstiness"),
            "dataset": values.get("dataset"),
        }

        # Combine into composite key
        composite = {
            "environment": env_values,
            "benchmark": benchmark_params,
        }
        return json.dumps(composite, sort_keys=True)


# Made with Bob
