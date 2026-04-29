# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Tests to verify that VLLM deployment parameters are correctly applied in the deployment YAML.
"""

import pytest
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
    VLLMDtype,
)


class TestDeploymentParameters:
    """Test suite for verifying VLLM deployment parameter configuration."""

    @pytest.fixture
    def base_deployment_params(self) -> dict:
        """Base parameters for creating a deployment."""
        return {
            "k8s_name": "test-vllm-deployment",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "gpu_type": "NVIDIA-A100-80GB-PCIe",
            "n_gpus": 1,
            "n_cpus": 8,
            "memory": "128Gi",
            "claim_name": None,  # Don't use PVC to avoid env issues in tests
        }

    def test_max_batch_tokens_not_set_by_default(
        self, base_deployment_params: dict
    ) -> None:
        """Test that max_batch_tokens is NOT set by default."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        # Extract the args from the container
        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        # Verify max_batch_tokens is NOT in args when not explicitly provided
        assert "--max-num-batched-tokens" not in args

    def test_max_batch_tokens_custom(self, base_deployment_params: dict) -> None:
        """Test that max_batch_tokens is correctly set with custom value."""
        custom_max_batch_tokens = 32768
        deployment_yaml = ComponentsYaml.deployment_yaml(
            **base_deployment_params, max_batch_tokens=custom_max_batch_tokens
        )

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        assert "--max-num-batched-tokens" in args
        idx = args.index("--max-num-batched-tokens")
        assert args[idx + 1] == str(custom_max_batch_tokens)

    def test_gpu_memory_utilization_not_set_by_default(
        self, base_deployment_params: dict
    ) -> None:
        """Test that gpu_memory_utilization is NOT set by default."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        # gpu_memory_utilization should NOT be in args when not explicitly provided
        assert "--gpu-memory-utilization" not in args

    def test_gpu_memory_utilization_custom(self, base_deployment_params: dict) -> None:
        """Test that gpu_memory_utilization is correctly set with custom value."""
        custom_gpu_memory = 0.85
        deployment_yaml = ComponentsYaml.deployment_yaml(
            **base_deployment_params, gpu_memory_utilization=custom_gpu_memory
        )

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        assert "--gpu-memory-utilization" in args
        idx = args.index("--gpu-memory-utilization")
        assert args[idx + 1] == str(custom_gpu_memory)

    def test_dtype_default(self, base_deployment_params: dict) -> None:
        """Test that dtype is correctly set with default value (auto)."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        assert "--dtype" in args
        idx = args.index("--dtype")
        assert args[idx + 1] == "auto"

    @pytest.mark.parametrize(
        ("dtype_value", "expected_str"),
        [
            (VLLMDtype.AUTO, "auto"),
            (VLLMDtype.HALF, "half"),
            (VLLMDtype.FLOAT16, "float16"),
            (VLLMDtype.BFLOAT16, "bfloat16"),
            (VLLMDtype.FLOAT, "float"),
            (VLLMDtype.FLOAT32, "float32"),
        ],
    )
    def test_dtype_all_values(
        self, base_deployment_params: dict, dtype_value: VLLMDtype, expected_str: str
    ) -> None:
        """Test that all dtype enum values are correctly applied."""
        deployment_yaml = ComponentsYaml.deployment_yaml(
            **base_deployment_params, dtype=dtype_value
        )

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        assert "--dtype" in args
        idx = args.index("--dtype")
        assert args[idx + 1] == expected_str

    def test_cpu_offload_not_set_by_default(self, base_deployment_params: dict) -> None:
        """Test that cpu_offload is NOT set by default."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        # cpu_offload should NOT be in args when not explicitly provided
        assert "--cpu-offload-gb" not in args

    def test_cpu_offload_custom(self, base_deployment_params: dict) -> None:
        """Test that cpu_offload is correctly set with custom value."""
        custom_cpu_offload = 16
        deployment_yaml = ComponentsYaml.deployment_yaml(
            **base_deployment_params, cpu_offload=custom_cpu_offload
        )

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        assert "--cpu-offload-gb" in args
        idx = args.index("--cpu-offload-gb")
        assert args[idx + 1] == str(custom_cpu_offload)

    def test_max_num_seq_not_set_by_default(self, base_deployment_params: dict) -> None:
        """Test that max_num_seq is NOT set by default."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        # max_num_seq should NOT be in args when not explicitly provided
        assert "--max-num-seq" not in args

    def test_max_num_seq_custom(self, base_deployment_params: dict) -> None:
        """Test that max_num_seq is correctly set with custom value."""
        custom_max_num_seq = 512
        deployment_yaml = ComponentsYaml.deployment_yaml(
            **base_deployment_params, max_num_seq=custom_max_num_seq
        )

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        assert "--max-num-seq" in args
        idx = args.index("--max-num-seq")
        assert args[idx + 1] == str(custom_max_num_seq)

    def test_all_parameters_together(self, base_deployment_params: dict) -> None:
        """Test that all parameters can be set together correctly."""
        custom_params = {
            "max_batch_tokens": 8192,
            "gpu_memory_utilization": 0.75,
            "dtype": VLLMDtype.BFLOAT16,
            "cpu_offload": 8,
            "max_num_seq": 128,
        }

        deployment_yaml = ComponentsYaml.deployment_yaml(
            **base_deployment_params, **custom_params
        )

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        # Verify all parameters are present with correct values
        assert "--max-num-batched-tokens" in args
        idx = args.index("--max-num-batched-tokens")
        assert args[idx + 1] == "8192"

        assert "--gpu-memory-utilization" in args
        idx = args.index("--gpu-memory-utilization")
        assert args[idx + 1] == "0.75"

        assert "--dtype" in args
        idx = args.index("--dtype")
        assert args[idx + 1] == "bfloat16"

        assert "--cpu-offload-gb" in args
        idx = args.index("--cpu-offload-gb")
        assert args[idx + 1] == "8"

        assert "--max-num-seq" in args
        idx = args.index("--max-num-seq")
        assert args[idx + 1] == "128"

    def test_parameters_order_in_args(self, base_deployment_params: dict) -> None:
        """Test that parameters appear in the expected order in args."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        # Verify the model is first
        assert args[0] == base_deployment_params["model"]

        # Verify required VLLM parameters are present
        assert "--dtype" in args

        # Verify optional parameters are NOT present by default (including tensor-parallel-size for single GPU)
        assert "--max-num-batched-tokens" not in args
        assert "--gpu-memory-utilization" not in args
        assert "--cpu-offload-gb" not in args
        assert "--max-num-seq" not in args
        assert "--tensor-parallel-size" not in args  # Not set for single GPU (n_gpus=1)

    def test_tensor_parallel_size_set_for_multi_gpu(self) -> None:
        """Test that tensor-parallel-size is set when n_gpus > 1."""
        n_gpus = 4
        deployment_yaml = ComponentsYaml.deployment_yaml(
            k8s_name="test-vllm-deployment",
            model="meta-llama/Llama-3.1-8B-Instruct",
            n_gpus=n_gpus,
        )

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        assert "--tensor-parallel-size" in args
        idx = args.index("--tensor-parallel-size")
        assert args[idx + 1] == str(n_gpus)

    def test_tensor_parallel_size_not_set_for_single_gpu(
        self, base_deployment_params: dict
    ) -> None:
        """Test that tensor-parallel-size is NOT set for single GPU (n_gpus=1)."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        # tensor-parallel-size should NOT be set for single GPU
        assert "--tensor-parallel-size" not in args

    def test_deployment_metadata_correct(self, base_deployment_params: dict) -> None:
        """Test that deployment metadata is correctly set."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        # Verify metadata
        assert deployment_yaml["metadata"]["name"] == base_deployment_params["k8s_name"]
        assert (
            deployment_yaml["metadata"]["labels"]["app.kubernetes.io/instance"]
            == base_deployment_params["k8s_name"]
        )

    def test_resource_limits_match_requests(self, base_deployment_params: dict) -> None:
        """Test that resource limits match resource requests."""
        deployment_yaml = ComponentsYaml.deployment_yaml(**base_deployment_params)

        container = deployment_yaml["spec"]["template"]["spec"]["containers"][0]
        resources = container["resources"]

        # Verify CPU
        assert resources["requests"]["cpu"] == str(base_deployment_params["n_cpus"])
        assert resources["limits"]["cpu"] == str(base_deployment_params["n_cpus"])

        # Verify memory
        assert resources["requests"]["memory"] == base_deployment_params["memory"]
        assert resources["limits"]["memory"] == base_deployment_params["memory"]

        # Verify GPU
        assert resources["requests"]["nvidia.com/gpu"] == str(
            base_deployment_params["n_gpus"]
        )
        assert resources["limits"]["nvidia.com/gpu"] == str(
            base_deployment_params["n_gpus"]
        )


# Made with Bob
