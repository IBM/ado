# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for build_components module, specifically renderer_num_workers handling."""

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
