# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for build_components module, specifically renderer_num_workers handling."""

from typing import Any

from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
)


class TestRendererNumWorkersInVllmArgs:
    """Test that renderer_num_workers is correctly handled in vLLM serve args."""

    def test_renderer_num_workers_zero_not_in_args(self) -> None:
        """When renderer_num_workers=0, --renderer-num-workers should NOT be in args."""
        # Create a minimal deployment spec
        result = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            n_gpus=1,
            gpu_type="nvidia-tesla-t4",
            n_cpus=4,
            memory="16Gi",
            max_num_seq=256,
            renderer_num_workers=0,  # Explicitly set to 0
        )

        # Extract the vLLM serve args from the deployment
        containers = result["spec"]["template"]["spec"]["containers"]
        vllm_container = next(c for c in containers if c["name"] == "vllm")
        args = vllm_container["args"]

        # Verify --renderer-num-workers is NOT in the args
        assert "--renderer-num-workers" not in args, (
            f"--renderer-num-workers should not be in args when renderer_num_workers=0, "
            f"but found in: {args}"
        )

    def test_renderer_num_workers_none_not_in_args(self) -> None:
        """When renderer_num_workers=None, --renderer-num-workers should NOT be in args."""
        result = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            n_gpus=1,
            gpu_type="nvidia-tesla-t4",
            n_cpus=4,
            memory="16Gi",
            max_num_seq=256,
            renderer_num_workers=None,  # Explicitly set to None
        )

        containers = result["spec"]["template"]["spec"]["containers"]
        vllm_container = next(c for c in containers if c["name"] == "vllm")
        args = vllm_container["args"]

        assert "--renderer-num-workers" not in args, (
            f"--renderer-num-workers should not be in args when renderer_num_workers=None, "
            f"but found in: {args}"
        )

    def test_renderer_num_workers_positive_in_args(self) -> None:
        """When renderer_num_workers>0, --renderer-num-workers SHOULD be in args."""
        result = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            n_gpus=1,
            gpu_type="nvidia-tesla-t4",
            n_cpus=4,
            memory="16Gi",
            max_num_seq=256,
            renderer_num_workers=32,  # Positive value
        )

        containers = result["spec"]["template"]["spec"]["containers"]
        vllm_container = next(c for c in containers if c["name"] == "vllm")
        args = vllm_container["args"]

        # Verify --renderer-num-workers IS in the args
        assert "--renderer-num-workers" in args, (
            f"--renderer-num-workers should be in args when renderer_num_workers=32, "
            f"but not found in: {args}"
        )

        # Verify the value is correct
        idx = args.index("--renderer-num-workers")
        assert args[idx + 1] == "32", (
            f"Expected renderer_num_workers value to be '32', got '{args[idx + 1]}'"
        )


class TestAgentFlagsInVllmArgs:
    """Test that agent-related vLLM flags are correctly handled in vLLM serve args."""

    def _base_yaml(self, **kwargs: Any) -> list:  # noqa: ANN401
        """Return vllm serve args from a minimal deployment spec."""
        result = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            n_gpus=1,
            gpu_type="nvidia-tesla-t4",
            n_cpus=4,
            memory="16Gi",
            max_num_seq=256,
            **kwargs,
        )
        containers = result["spec"]["template"]["spec"]["containers"]
        vllm_container = next(c for c in containers if c["name"] == "vllm")
        return vllm_container["args"]

    def test_reasoning_parser_in_args(self) -> None:
        """When reasoning_parser is set, --reasoning-parser <value> should be in args."""
        args = self._base_yaml(reasoning_parser="qwen3")
        assert "--reasoning-parser" in args, f"--reasoning-parser not found in: {args}"
        idx = args.index("--reasoning-parser")
        assert args[idx + 1] == "qwen3", f"Expected 'qwen3', got '{args[idx + 1]}'"

    def test_reasoning_parser_none_not_in_args(self) -> None:
        """When reasoning_parser is None, --reasoning-parser should NOT be in args."""
        args = self._base_yaml(reasoning_parser=None)
        assert "--reasoning-parser" not in args, (
            f"--reasoning-parser found unexpectedly in: {args}"
        )

    def test_tool_call_parser_in_args(self) -> None:
        """When tool_call_parser is set, --tool-call-parser <value> should be in args."""
        args = self._base_yaml(tool_call_parser="qwen3_coder")
        assert "--tool-call-parser" in args, f"--tool-call-parser not found in: {args}"
        idx = args.index("--tool-call-parser")
        assert args[idx + 1] == "qwen3_coder", (
            f"Expected 'qwen3_coder', got '{args[idx + 1]}'"
        )

    def test_tool_call_parser_none_not_in_args(self) -> None:
        """When tool_call_parser is None, --tool-call-parser should NOT be in args."""
        args = self._base_yaml(tool_call_parser=None)
        assert "--tool-call-parser" not in args, (
            f"--tool-call-parser found unexpectedly in: {args}"
        )

    def test_language_model_only_in_args(self) -> None:
        """When language_model_only=True, --language-model-only flag should be in args."""
        args = self._base_yaml(language_model_only=True)
        assert "--language-model-only" in args, (
            f"--language-model-only not found in: {args}"
        )

    def test_language_model_only_false_not_in_args(self) -> None:
        """When language_model_only=False, --language-model-only should NOT be in args."""
        args = self._base_yaml(language_model_only=False)
        assert "--language-model-only" not in args, (
            f"--language-model-only found unexpectedly in: {args}"
        )

    def test_enable_auto_tool_choice_in_args(self) -> None:
        """When enable_auto_tool_choice=True, --enable-auto-tool-choice flag should be in args."""
        args = self._base_yaml(enable_auto_tool_choice=True)
        assert "--enable-auto-tool-choice" in args, (
            f"--enable-auto-tool-choice not found in: {args}"
        )

    def test_enable_auto_tool_choice_false_not_in_args(self) -> None:
        """When enable_auto_tool_choice=False, --enable-auto-tool-choice should NOT be in args."""
        args = self._base_yaml(enable_auto_tool_choice=False)
        assert "--enable-auto-tool-choice" not in args, (
            f"--enable-auto-tool-choice found unexpectedly in: {args}"
        )

    def test_max_model_len_in_args(self) -> None:
        """When max_model_len is set, --max-model-len <value> should be in args."""
        args = self._base_yaml(max_model_len=262144)
        assert "--max-model-len" in args, f"--max-model-len not found in: {args}"
        idx = args.index("--max-model-len")
        assert args[idx + 1] == "262144", f"Expected '262144', got '{args[idx + 1]}'"

    def test_max_model_len_none_not_in_args(self) -> None:
        """When max_model_len is None, --max-model-len should NOT be in args."""
        args = self._base_yaml(max_model_len=None)
        assert "--max-model-len" not in args, (
            f"--max-model-len found unexpectedly in: {args}"
        )


class TestNewServerParamsInVllmArgs:
    """Test that kv_cache_dtype and enable_prefix_caching are correctly handled."""

    def _base_yaml(self, **kwargs: Any) -> list:  # noqa: ANN401
        """Return vllm serve args from a minimal deployment spec."""
        result = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            n_gpus=1,
            gpu_type="nvidia-tesla-t4",
            n_cpus=4,
            memory="16Gi",
            max_num_seq=256,
            **kwargs,
        )
        containers = result["spec"]["template"]["spec"]["containers"]
        vllm_container = next(c for c in containers if c["name"] == "vllm")
        return vllm_container["args"]

    def test_kv_cache_dtype_in_args(self) -> None:
        """When kv_cache_dtype is set, --kv-cache-dtype <value> should be in args."""
        args = self._base_yaml(kv_cache_dtype="fp8")
        assert "--kv-cache-dtype" in args, f"--kv-cache-dtype not found in: {args}"
        idx = args.index("--kv-cache-dtype")
        assert args[idx + 1] == "fp8", f"Expected 'fp8', got '{args[idx + 1]}'"

    def test_kv_cache_dtype_none_not_in_args(self) -> None:
        """When kv_cache_dtype is None, --kv-cache-dtype should NOT be in args."""
        args = self._base_yaml(kv_cache_dtype=None)
        assert "--kv-cache-dtype" not in args, (
            f"--kv-cache-dtype found unexpectedly in: {args}"
        )

    def test_enable_prefix_caching_in_args(self) -> None:
        """When enable_prefix_caching=True, --enable-prefix-caching flag should be in args."""
        args = self._base_yaml(enable_prefix_caching=True)
        assert "--enable-prefix-caching" in args, (
            f"--enable-prefix-caching not found in: {args}"
        )

    def test_enable_prefix_caching_false_not_in_args(self) -> None:
        """When enable_prefix_caching=False, --enable-prefix-caching should NOT be in args."""
        args = self._base_yaml(enable_prefix_caching=False)
        assert "--enable-prefix-caching" not in args, (
            f"--enable-prefix-caching found unexpectedly in: {args}"
        )


class TestBuildEntityEnv:
    """Test that _build_entity_env produces distinct cache keys for each deployment-affecting parameter."""

    def _base_values(self) -> dict:
        """Return a minimal valid values dict with all required deployment parameters set."""
        return {
            "model": "test-model",
            "image": "vllm/vllm-openai:v0.14.0",
            "n_gpus": 1,
            "gpu_type": "NVIDIA-A100-80GB-PCIe",
            "n_cpus": 4,
            "memory": "16Gi",
            "max_batch_tokens": 16384,
            "gpu_memory_utilization": 0.9,
            "dtype": "auto",
            "cpu_offload": 0,
            "max_num_seq": 256,
            "renderer_num_workers": None,
            "reasoning_parser": None,
            "tool_call_parser": None,
            "language_model_only": 0,
            "enable_auto_tool_choice": 0,
            "max_model_len": None,
            "kv_cache_dtype": None,
            "enable_prefix_caching": 0,
        }

    def test_kv_cache_dtype_produces_distinct_key(self) -> None:
        """Entities differing only in kv_cache_dtype must not share a deployment environment."""
        from ado_actuators.vllm_performance.experiment_executor import _build_entity_env

        with_value = {**self._base_values(), "kv_cache_dtype": "fp8"}
        without_value = {**self._base_values(), "kv_cache_dtype": None}
        assert _build_entity_env(with_value) != _build_entity_env(without_value), (
            "kv_cache_dtype=fp8 and kv_cache_dtype=None must produce different cache keys"
        )

    def test_enable_prefix_caching_produces_distinct_key(self) -> None:
        """Entities differing only in enable_prefix_caching must not share a deployment environment."""
        from ado_actuators.vllm_performance.experiment_executor import _build_entity_env

        with_value = {**self._base_values(), "enable_prefix_caching": 1}
        without_value = {**self._base_values(), "enable_prefix_caching": 0}
        assert _build_entity_env(with_value) != _build_entity_env(without_value), (
            "enable_prefix_caching=1 and enable_prefix_caching=0 must produce different cache keys"
        )
