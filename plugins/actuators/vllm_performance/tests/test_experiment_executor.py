# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for experiment_executor module functions."""

import json

import pytest
from ado_actuators.vllm_performance.cache_utils import CacheKeyBuilder
from ado_actuators.vllm_performance.version_utils import VLLMVersionChecker


@pytest.fixture
def base_env_values():
    """Base environment values for testing."""
    return {
        "model": "test-model",
        "n_gpus": "1",
        "gpu_type": "nvidia-a100",
        "n_cpus": "8",
        "memory": "32Gi",
        "max_batch_tokens": "4096",
        "gpu_memory_utilization": "0.9",
        "dtype": "auto",
        "cpu_offload": "0",
        "max_num_seq": "256",
    }


@pytest.fixture
def base_benchmark_values():
    """Base benchmark values for testing."""
    return {
        "num_prompts": "200",
        "request_rate": "32",
        "dataset": "random",
    }


class TestGetVllmVersionFromImageValue:
    """Test suite for version extraction from image values."""

    @pytest.mark.parametrize(
        "image_value,expected",
        [
            (["icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5", "0.18.0"], "0.18.0"),
            (["vllm/vllm-openai:v0.14.0", "0.14.0"], "0.14.0"),
            (["vllm/vllm-openai:latest", "0.21.0"], "0.21.0"),
            ("icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5", None),
            (["icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5"], None),
        ],
    )
    def test_version_extraction(self, image_value, expected) -> None:
        """Test version extraction from various image value formats."""
        version = VLLMVersionChecker.parse_version(image_value)
        assert version == expected


class TestBuildEntityEnv:
    """Test suite for environment definition building."""

    @pytest.mark.parametrize(
        "image,threadpool,renderer_workers,expected_threadpool,expected_workers",
        [
            (["icr.io/test/vllm:v0.18.0", "0.18.0"], "1", "64", 0, 0),
            (["icr.io/test/vllm:v0.21.0", "0.21.0"], "1", "64", 1, 64),
            (["icr.io/test/vllm:v0.21.0", "0.21.0"], "0", "64", 0, 0),
            ("icr.io/test/vllm:v0.18.0", "1", "64", 1, 64),
        ],
    )
    def test_threadpool_normalization(
        self,
        base_env_values,
        image,
        threadpool,
        renderer_workers,
        expected_threadpool,
        expected_workers,
    ) -> None:
        """Test threadpool and renderer_num_workers normalization."""
        values = {
            **base_env_values,
            "image": image,
            "threadpool": threadpool,
            "renderer_num_workers": renderer_workers,
        }

        result = CacheKeyBuilder.build_env_definition(values)
        result_dict = json.loads(result)

        assert result_dict["threadpool"] == expected_threadpool
        assert result_dict["renderer_num_workers"] == expected_workers

    def test_different_renderer_workers_same_env_vllm_0_18(
        self, base_env_values
    ) -> None:
        """Test different renderer_num_workers produce same env for vLLM < 0.20.0."""
        base = {
            **base_env_values,
            "image": ["icr.io/test/vllm:v0.18.0", "0.18.0"],
            "threadpool": "1",
        }

        envs = [
            CacheKeyBuilder.build_env_definition({**base, "renderer_num_workers": w})
            for w in ["32", "64", "128"]
        ]

        assert envs[0] == envs[1] == envs[2]
        assert json.loads(envs[0])["renderer_num_workers"] == 0


class TestBuildBenchmarkParamsKey:
    """Test suite for benchmark parameter extraction."""

    def test_includes_all_benchmark_parameters(self) -> None:
        """Test all benchmark parameters are included."""
        values = {
            "num_prompts": "100",
            "request_rate": "10",
            "max_concurrency": "5",
            "number_input_tokens": "50",
            "max_output_tokens": "100",
            "burstiness": "1.0",
            "dataset": "random",
        }

        cache_key = CacheKeyBuilder.build(values)
        benchmark = json.loads(cache_key)["benchmark"]

        assert benchmark["num_prompts"] == "100"
        assert benchmark["request_rate"] == "10"
        assert benchmark["max_concurrency"] == "5"
        assert benchmark["number_input_tokens"] == "50"
        assert benchmark["max_output_tokens"] == "100"
        assert benchmark["burstiness"] == "1.0"
        assert benchmark["dataset"] == "random"

    def test_handles_missing_values(self) -> None:
        """Test missing values are handled as None."""
        cache_key = CacheKeyBuilder.build({"num_prompts": "100"})
        benchmark = json.loads(cache_key)["benchmark"]

        assert benchmark["num_prompts"] == "100"
        assert benchmark["request_rate"] is None
        assert benchmark["max_concurrency"] is None

    def test_consistent_output_with_sorted_keys(self) -> None:
        """Test output is consistent with sorted keys."""
        values = {"dataset": "random", "num_prompts": "100", "request_rate": "10"}

        key1 = json.dumps(
            json.loads(CacheKeyBuilder.build(values))["benchmark"], sort_keys=True
        )
        key2 = json.dumps(
            json.loads(CacheKeyBuilder.build(values))["benchmark"], sort_keys=True
        )

        assert key1 == key2
        assert list(json.loads(key1).keys()) == sorted(json.loads(key1).keys())


class TestBuildCacheKey:
    """Test suite for complete cache key building."""

    def test_combines_environment_and_benchmark_params(
        self, base_env_values, base_benchmark_values
    ) -> None:
        """Test cache key includes both environment and benchmark sections."""
        values = {
            **base_env_values,
            "image": ["icr.io/test/vllm:v0.18.0", "0.18.0"],
            "threadpool": "1",
            "renderer_num_workers": "32",
            **base_benchmark_values,
        }

        result_dict = json.loads(CacheKeyBuilder.build(values))

        assert "environment" in result_dict
        assert "benchmark" in result_dict
        assert result_dict["environment"]["model"] == "test-model"
        assert result_dict["benchmark"]["num_prompts"] == "200"

    @pytest.mark.parametrize(
        "param,value1,value2",
        [
            ("num_prompts", "100", "200"),
            ("request_rate", "32", "64"),
        ],
    )
    def test_different_params_produce_different_keys(
        self, base_env_values, base_benchmark_values, param, value1, value2
    ) -> None:
        """Test different parameter values produce different cache keys."""
        base = {
            **base_env_values,
            "image": ["icr.io/test/vllm:v0.20.1", "0.20.1"],
            "threadpool": "1",
            "renderer_num_workers": "32",
            **base_benchmark_values,
        }

        key1 = CacheKeyBuilder.build({**base, param: value1})
        key2 = CacheKeyBuilder.build({**base, param: value2})

        assert key1 != key2

    def test_same_params_produce_same_key(
        self, base_env_values, base_benchmark_values
    ) -> None:
        """Test identical parameters produce identical cache keys."""
        values = {
            **base_env_values,
            "image": ["icr.io/test/vllm:v0.20.1", "0.20.1"],
            "threadpool": "1",
            "renderer_num_workers": "32",
            **base_benchmark_values,
        }

        assert CacheKeyBuilder.build(values) == CacheKeyBuilder.build(values)

    def test_vllm_0_18_same_key_different_renderer_workers(
        self, base_env_values, base_benchmark_values
    ) -> None:
        """Test vLLM 0.18.0 produces same key for different renderer_num_workers."""
        base = {
            **base_env_values,
            "image": ["icr.io/test/vllm:v0.18.0", "0.18.0"],
            "threadpool": "1",
            **base_benchmark_values,
        }

        key_32 = CacheKeyBuilder.build({**base, "renderer_num_workers": "32"})
        key_64 = CacheKeyBuilder.build({**base, "renderer_num_workers": "64"})

        assert key_32 == key_64
        assert json.loads(key_32)["environment"]["renderer_num_workers"] == 0

    def test_vllm_0_20_different_key_different_renderer_workers(
        self, base_env_values, base_benchmark_values
    ) -> None:
        """Test vLLM 0.20.1 produces different keys for different renderer_num_workers."""
        base = {
            **base_env_values,
            "image": ["icr.io/test/vllm:v0.20.1", "0.20.1"],
            "threadpool": "1",
            **base_benchmark_values,
        }

        key_32 = CacheKeyBuilder.build({**base, "renderer_num_workers": "32"})
        key_64 = CacheKeyBuilder.build({**base, "renderer_num_workers": "64"})

        assert key_32 != key_64
        assert json.loads(key_32)["environment"]["renderer_num_workers"] == 32
        assert json.loads(key_64)["environment"]["renderer_num_workers"] == 64
