# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Unit tests for experiment_executor module functions.
Tests version extraction from image property values.
"""

import json

from ado_actuators.vllm_performance.experiment_executor import (
    _build_benchmark_params_key,
    _build_cache_key,
    _build_entity_env,
    _get_vllm_version_from_image_value,
)


class TestGetVllmVersionFromImageValue:
    """Test suite for _get_vllm_version_from_image_value function"""

    def test_version_extraction_from_list_value(self) -> None:
        """Test extracting vLLM version from list image value"""
        image_value = [
            "icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5",
            "0.18.0",
        ]

        version = _get_vllm_version_from_image_value(image_value)
        assert version == "0.18.0"

    def test_version_extraction_from_another_list_value(self) -> None:
        """Test extracting vLLM version from another list image value"""
        image_value = [
            "vllm/vllm-openai:v0.14.0",
            "0.14.0",
        ]

        version = _get_vllm_version_from_image_value(image_value)
        assert version == "0.14.0"

    def test_version_extraction_returns_none_for_string_value(self) -> None:
        """Test that None is returned when image value is a string (backward compatibility)"""
        image_value = "icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5"

        version = _get_vllm_version_from_image_value(image_value)
        assert version is None

    def test_version_extraction_returns_none_for_list_without_version(self) -> None:
        """Test that None is returned when list has only one element (no version)"""
        image_value = [
            "icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5",
        ]

        version = _get_vllm_version_from_image_value(image_value)
        assert version is None

    def test_version_extraction_with_latest_tag(self) -> None:
        """Test extracting version for latest tag"""
        image_value = [
            "vllm/vllm-openai:latest",
            "0.21.0",
        ]

        version = _get_vllm_version_from_image_value(image_value)
        assert version == "0.21.0"


class TestBuildEntityEnv:
    """Test suite for _build_entity_env function"""

    def test_renderer_num_workers_normalized_when_vllm_version_less_than_0_20_0(
        self,
    ) -> None:
        """Test that renderer_num_workers is normalized to 0 when vLLM < 0.20.0"""
        # Test with vLLM 0.18.0 (< 0.20.0)
        values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.18.0", "0.18.0"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "renderer_num_workers": "64",  # Should be normalized to 0
        }

        result = _build_entity_env(values)
        result_dict = json.loads(result)

        # Both threadpool and renderer_num_workers should be 0
        assert result_dict["threadpool"] == 0
        assert result_dict["renderer_num_workers"] == 0

    def test_renderer_num_workers_preserved_when_vllm_version_greater_than_0_20_0(
        self,
    ) -> None:
        """Test that renderer_num_workers is preserved when vLLM >= 0.20.0"""
        # Test with vLLM 0.21.0 (>= 0.20.0)
        values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.21.0", "0.21.0"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "renderer_num_workers": "64",
        }

        result = _build_entity_env(values)
        result_dict = json.loads(result)

        # Both should be preserved
        assert result_dict["threadpool"] == 1
        assert result_dict["renderer_num_workers"] == 64

    def test_renderer_num_workers_normalized_when_threadpool_disabled_by_user(
        self,
    ) -> None:
        """Test that renderer_num_workers is normalized to 0 when user disables threadpool"""
        # Test with vLLM 0.21.0 but threadpool=0
        values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.21.0", "0.21.0"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "0",  # User explicitly disabled
            "renderer_num_workers": "64",  # Should be normalized to 0
        }

        result = _build_entity_env(values)
        result_dict = json.loads(result)

        # Both should be 0
        assert result_dict["threadpool"] == 0
        assert result_dict["renderer_num_workers"] == 0

    def test_different_renderer_num_workers_same_env_when_vllm_less_than_0_20_0(
        self,
    ) -> None:
        """Test that different renderer_num_workers values produce same env when vLLM < 0.20.0"""
        base_values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.18.0", "0.18.0"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
        }

        # Test with different renderer_num_workers values
        values_32 = {**base_values, "renderer_num_workers": "32"}
        values_64 = {**base_values, "renderer_num_workers": "64"}
        values_128 = {**base_values, "renderer_num_workers": "128"}

        env_32 = _build_entity_env(values_32)
        env_64 = _build_entity_env(values_64)
        env_128 = _build_entity_env(values_128)

        # All should produce the same environment definition
        assert env_32 == env_64 == env_128

        # Verify they all have renderer_num_workers=0
        result_dict = json.loads(env_32)
        assert result_dict["renderer_num_workers"] == 0

    def test_backward_compatibility_with_string_image(self) -> None:
        """Test backward compatibility when image is a string (no version info)"""
        values = {
            "model": "test-model",
            "image": "icr.io/test/vllm:v0.18.0",  # String, no version info
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "renderer_num_workers": "64",
        }

        result = _build_entity_env(values)
        result_dict = json.loads(result)

        # Should assume threadpool is supported (backward compatible)
        assert result_dict["threadpool"] == 1
        assert result_dict["renderer_num_workers"] == 64


class TestBuildBenchmarkParamsKey:
    """Test suite for _build_benchmark_params_key function"""

    def test_includes_all_benchmark_parameters(self) -> None:
        """Test that all benchmark parameters are included in the key"""
        values = {
            "num_prompts": "100",
            "request_rate": "10",
            "max_concurrency": "5",
            "number_input_tokens": "50",
            "max_output_tokens": "100",
            "burstiness": "1.0",
            "dataset": "random",
        }

        result = _build_benchmark_params_key(values)
        result_dict = json.loads(result)

        assert result_dict["num_prompts"] == "100"
        assert result_dict["request_rate"] == "10"
        assert result_dict["max_concurrency"] == "5"
        assert result_dict["number_input_tokens"] == "50"
        assert result_dict["max_output_tokens"] == "100"
        assert result_dict["burstiness"] == "1.0"
        assert result_dict["dataset"] == "random"

    def test_handles_missing_values(self) -> None:
        """Test that missing values are handled as None"""
        values = {
            "num_prompts": "100",
            # Other parameters missing
        }

        result = _build_benchmark_params_key(values)
        result_dict = json.loads(result)

        assert result_dict["num_prompts"] == "100"
        assert result_dict["request_rate"] is None
        assert result_dict["max_concurrency"] is None
        assert result_dict["dataset"] is None

    def test_consistent_output_with_sorted_keys(self) -> None:
        """Test that output is consistent (keys are sorted)"""
        values = {
            "dataset": "random",
            "num_prompts": "100",
            "request_rate": "10",
        }

        result1 = _build_benchmark_params_key(values)
        result2 = _build_benchmark_params_key(values)

        # Should produce identical output
        assert result1 == result2

        # Verify keys are sorted in JSON
        result_dict = json.loads(result1)
        keys = list(result_dict.keys())
        assert keys == sorted(keys)


class TestBuildCacheKey:
    """Test suite for _build_cache_key function"""

    def test_combines_environment_and_benchmark_params(self) -> None:
        """Test that cache key includes both environment and benchmark parameters"""
        values = {
            # Environment params
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.18.0", "0.18.0"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "renderer_num_workers": "32",
            # Benchmark params
            "num_prompts": "200",
            "request_rate": "32",
            "dataset": "random",
        }

        result = _build_cache_key(values)
        result_dict = json.loads(result)

        # Should have both environment and benchmark sections
        assert "environment" in result_dict
        assert "benchmark" in result_dict

        # Check environment section
        env = result_dict["environment"]
        assert env["model"] == "test-model"
        assert env["n_gpus"] == "1"

        # Check benchmark section
        benchmark = result_dict["benchmark"]
        assert benchmark["num_prompts"] == "200"
        assert benchmark["request_rate"] == "32"
        assert benchmark["dataset"] == "random"

    def test_different_benchmark_params_produce_different_keys(self) -> None:
        """Test that different benchmark parameters produce different cache keys"""
        base_values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.20.1", "0.20.1"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "renderer_num_workers": "32",
        }

        # Same environment, different num_prompts
        values1 = {**base_values, "num_prompts": "100", "request_rate": "32"}
        values2 = {**base_values, "num_prompts": "200", "request_rate": "32"}

        key1 = _build_cache_key(values1)
        key2 = _build_cache_key(values2)

        # Different benchmark params should produce different keys
        assert key1 != key2

    def test_same_params_produce_same_key(self) -> None:
        """Test that identical parameters produce identical cache keys"""
        values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.20.1", "0.20.1"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "renderer_num_workers": "32",
            "num_prompts": "200",
            "request_rate": "32",
            "dataset": "random",
        }

        key1 = _build_cache_key(values)
        key2 = _build_cache_key(values)

        # Identical params should produce identical keys
        assert key1 == key2

    def test_cache_key_differentiates_on_request_rate(self) -> None:
        """Test that different request_rate values produce different cache keys"""
        base_values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.18.0", "0.18.0"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "renderer_num_workers": "32",
            "num_prompts": "200",
            "dataset": "random",
        }

        # Same everything except request_rate
        values_rate_32 = {**base_values, "request_rate": "32"}
        values_rate_64 = {**base_values, "request_rate": "64"}

        key_32 = _build_cache_key(values_rate_32)
        key_64 = _build_cache_key(values_rate_64)

        # Different request rates should produce different keys
        assert key_32 != key_64

    def test_vllm_0_18_same_cache_key_for_different_renderer_num_workers(self) -> None:
        """
        Test that for vLLM 0.18.0, different renderer_num_workers values produce
        the SAME cache key (because threadpool is not supported and normalized to 0)
        """
        base_values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.18.0", "0.18.0"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "num_prompts": "200",
            "request_rate": "32",
            "dataset": "random",
        }

        # Different renderer_num_workers values
        values_32 = {**base_values, "renderer_num_workers": "32"}
        values_64 = {**base_values, "renderer_num_workers": "64"}

        key_32 = _build_cache_key(values_32)
        key_64 = _build_cache_key(values_64)

        # For vLLM 0.18.0, both should produce the same key
        # because renderer_num_workers is normalized to 0 in the environment
        assert key_32 == key_64

        # Verify the environment section has renderer_num_workers=0
        result_dict = json.loads(key_32)
        assert result_dict["environment"]["renderer_num_workers"] == 0

    def test_vllm_0_20_different_cache_key_for_different_renderer_num_workers(
        self,
    ) -> None:
        """
        Test that for vLLM 0.20.1, different renderer_num_workers values produce
        DIFFERENT cache keys (because threadpool is supported)
        """
        base_values = {
            "model": "test-model",
            "image": ["icr.io/test/vllm:v0.20.1", "0.20.1"],
            "n_gpus": "1",
            "gpu_type": "nvidia-a100",
            "n_cpus": "8",
            "memory": "32Gi",
            "max_batch_tokens": "4096",
            "gpu_memory_utilization": "0.9",
            "dtype": "auto",
            "cpu_offload": "0",
            "max_num_seq": "256",
            "threadpool": "1",
            "num_prompts": "200",
            "request_rate": "32",
            "dataset": "random",
        }

        # Different renderer_num_workers values
        values_32 = {**base_values, "renderer_num_workers": "32"}
        values_64 = {**base_values, "renderer_num_workers": "64"}

        key_32 = _build_cache_key(values_32)
        key_64 = _build_cache_key(values_64)

        # For vLLM 0.20.1, should produce different keys
        assert key_32 != key_64

        # Verify the environment sections have different renderer_num_workers
        result_dict_32 = json.loads(key_32)
        result_dict_64 = json.loads(key_64)
        assert result_dict_32["environment"]["renderer_num_workers"] == 32
        assert result_dict_64["environment"]["renderer_num_workers"] == 64


# Made with Bob
