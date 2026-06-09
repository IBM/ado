# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for cache key building utilities."""

import json
from typing import Any

import pytest
from ado_actuators.vllm_performance.cache_utils import CacheKeyBuilder


class TestCacheKeyBuilder:
    """Tests for CacheKeyBuilder class."""

    @pytest.fixture
    def base_values(self) -> dict[str, Any]:
        """Base values for testing."""
        return {
            "model": "meta-llama/Llama-2-7b-hf",
            "image": ["vllm/vllm-openai:v0.20.1", "0.20.1"],
            "n_gpus": "1",
            "gpu_type": "nvidia-l4",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "num_prompts": "100",
            "request_rate": "10",
            "max_concurrency": "50",
            "number_input_tokens": "128",
            "max_output_tokens": "256",
            "burstiness": "1.0",
            "dataset": "random",
        }

    def test_build_returns_json_string(self, base_values: dict[str, Any]) -> None:
        """Test that build returns a valid JSON string."""
        cache_key = CacheKeyBuilder.build(base_values)
        assert isinstance(cache_key, str)
        parsed = json.loads(cache_key)
        assert "environment" in parsed
        assert "benchmark" in parsed

    def test_same_values_produce_same_key(self, base_values: dict[str, Any]) -> None:
        """Test that identical values produce identical cache keys."""
        key1 = CacheKeyBuilder.build(base_values)
        key2 = CacheKeyBuilder.build(base_values.copy())
        assert key1 == key2

    def test_different_env_params_produce_different_keys(
        self, base_values: dict[str, Any]
    ) -> None:
        """Test that different environment parameters produce different keys."""
        key1 = CacheKeyBuilder.build(base_values)

        modified_values = base_values.copy()
        modified_values["n_gpus"] = "2"
        key2 = CacheKeyBuilder.build(modified_values)

        assert key1 != key2

    def test_different_benchmark_params_produce_different_keys(
        self, base_values: dict[str, Any]
    ) -> None:
        """Test that different benchmark parameters produce different keys."""
        key1 = CacheKeyBuilder.build(base_values)

        modified_values = base_values.copy()
        modified_values["num_prompts"] = "200"
        key2 = CacheKeyBuilder.build(modified_values)

        assert key1 != key2

    def test_image_list_extraction(self, base_values: dict[str, Any]) -> None:
        """Test that image URL is correctly extracted from list."""
        base_values["image"] = ["vllm/vllm-openai:v0.20.1", "0.20.1"]
        cache_key = CacheKeyBuilder.build(base_values)
        parsed = json.loads(cache_key)

        assert parsed["environment"]["image"] == "vllm/vllm-openai:v0.20.1"

    def test_image_string_backward_compatibility(
        self, base_values: dict[str, Any]
    ) -> None:
        """Test backward compatibility with string image values."""
        base_values["image"] = "vllm/vllm-openai:v0.20.1"
        cache_key = CacheKeyBuilder.build(base_values)
        parsed = json.loads(cache_key)

        assert parsed["environment"]["image"] == "vllm/vllm-openai:v0.20.1"

    def test_build_env_definition(self, base_values: dict[str, Any]) -> None:
        """Test environment definition building."""
        env_def = CacheKeyBuilder.build_env_definition(base_values)
        parsed = json.loads(env_def)

        assert parsed["model"] == "meta-llama/Llama-2-7b-hf"
        assert parsed["n_gpus"] == "1"
        assert parsed["image"] == "vllm/vllm-openai:v0.20.1"
