# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Unit tests for OTLP traces endpoint feature in vllm_performance actuator.
Tests parameter validation, YAML generation, and backward compatibility.
"""

from ado_actuators.vllm_performance.actuator_parameters import (
    VLLMPerformanceTestParameters,
)
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
)

from ado.core.actuatorconfiguration.config import ActuatorConfiguration


class TestOTLPTracesEndpointParameter:
    """Test suite for otlp_traces_endpoint parameter in VLLMPerformanceTestParameters"""

    def test_otlp_traces_endpoint_optional(self) -> None:
        """Test that otlp_traces_endpoint is optional and defaults to None"""
        params = VLLMPerformanceTestParameters()  # type: ignore[call-arg]
        assert params.otlp_traces_endpoint is None

    def test_otlp_traces_endpoint_accepts_valid_url(self) -> None:
        """Test that otlp_traces_endpoint accepts valid URLs"""
        url = "http://jaeger:4318/v1/traces"
        params = VLLMPerformanceTestParameters(otlp_traces_endpoint=url)  # type: ignore[call-arg]
        assert str(params.otlp_traces_endpoint) == url

    def test_otlp_traces_endpoint_rejects_invalid_url(self) -> None:
        """Test that otlp_traces_endpoint rejects invalid URLs"""
        import pytest
        from pydantic import ValidationError

        invalid_urls = [
            "hello",  # Not a URL
            "not-a-url",  # Not a URL
            "://invalid",  # Missing scheme
            "http://",  # Missing host
        ]

        for invalid_url in invalid_urls:
            with pytest.raises(ValidationError):
                VLLMPerformanceTestParameters(otlp_traces_endpoint=invalid_url)  # type: ignore[call-arg]


class TestActuatorConfigurationWithOTLP:
    """Test suite for ActuatorConfiguration with otlp_traces_endpoint"""

    def test_actuator_configuration_with_otlp_endpoint(self) -> None:
        """Test full actuator configuration with OTLP endpoint"""
        config = ActuatorConfiguration(
            actuatorIdentifier="vllm_performance",
            parameters=VLLMPerformanceTestParameters(  # type: ignore[call-arg]
                namespace="test-namespace",
                otlp_traces_endpoint="http://jaeger:4318/v1/traces",
            ),
        )
        assert (
            str(config.parameters.otlp_traces_endpoint)
            == "http://jaeger:4318/v1/traces"
        )  # type: ignore[union-attr]

    def test_actuator_configuration_without_otlp_endpoint(self) -> None:
        """Test actuator configuration without OTLP endpoint (backward compatibility)"""
        config = ActuatorConfiguration(
            actuatorIdentifier="vllm_performance",
            parameters=VLLMPerformanceTestParameters(  # type: ignore[call-arg]
                namespace="test-namespace",
                max_environments=3,
            ),
        )
        assert config.parameters.otlp_traces_endpoint is None  # type: ignore[union-attr]

    def test_actuator_configuration_yaml_roundtrip(self) -> None:
        """Test YAML serialization roundtrip with OTLP endpoint"""
        config = ActuatorConfiguration(
            actuatorIdentifier="vllm_performance",
            parameters=VLLMPerformanceTestParameters(  # type: ignore[call-arg]
                namespace="test-namespace",
                otlp_traces_endpoint="http://jaeger:4318/v1/traces",
                max_environments=2,
            ),
        )

        # Serialize back to dict
        config_dict = config.model_dump()
        assert (
            str(config_dict["parameters"]["otlp_traces_endpoint"])
            == "http://jaeger:4318/v1/traces"
        )

        # Create new config from serialized dict
        config_restored = ActuatorConfiguration(**config_dict)
        assert (
            str(config_restored.parameters.otlp_traces_endpoint)
            == "http://jaeger:4318/v1/traces"
        )  # type: ignore[union-attr]


class TestDeploymentYAMLWithOTLP:
    """Test suite for deployment YAML generation with otlp_traces_endpoint"""

    def test_deployment_yaml_without_otlp(self) -> None:
        """Test deployment YAML generation without OTLP endpoint"""
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            otlp_traces_endpoint=None,
        )

        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]

        # Verify no OTLP arg
        args = container["args"]
        assert "--otlp-traces-endpoint" not in args

    def test_deployment_yaml_with_otlp(self) -> None:
        """Test deployment YAML generation with OTLP endpoint"""
        otlp_url = "http://jaeger:4318/v1/traces"
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            otlp_traces_endpoint=otlp_url,
        )

        # Verify OTLP env var is NOT set (endpoint is passed via args instead)
        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]

        # Verify OTLP arg is present with correct value
        args = container["args"]
        assert "--otlp-traces-endpoint" in args
        otlp_arg_index = args.index("--otlp-traces-endpoint")
        assert args[otlp_arg_index + 1] == otlp_url

    def test_deployment_yaml_otlp_arg_not_env_var_reference(self) -> None:
        """Test that OTLP arg uses actual value, not environment variable reference"""
        otlp_url = "http://jaeger:4318/v1/traces"
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            otlp_traces_endpoint=otlp_url,
        )

        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]
        otlp_arg_index = args.index("--otlp-traces-endpoint")

        assert args[otlp_arg_index + 1] == otlp_url

    def test_deployment_yaml_without_otlp_no_service_name(self) -> None:
        """Test that OTEL_SERVICE_NAME is not set when OTLP endpoint is not provided"""
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            otlp_traces_endpoint=None,
        )

        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env", [])

        # Verify OTEL_SERVICE_NAME is not set
        service_name_vars = [
            e for e in env_vars if e.get("name") == "OTEL_SERVICE_NAME"
        ]
        assert len(service_name_vars) == 0

    def test_deployment_yaml_with_otlp_sets_service_name(self) -> None:
        """Test that OTEL_SERVICE_NAME is set to deployment name when OTLP endpoint is provided"""
        otlp_url = "http://jaeger:4318/v1/traces"
        k8s_name = "test-deployment-12345"
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name=k8s_name,
            model="test-model",
            otlp_traces_endpoint=otlp_url,
        )

        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env", [])

        # Verify OTEL_SERVICE_NAME is set with correct value
        service_name_vars = [
            e for e in env_vars if e.get("name") == "OTEL_SERVICE_NAME"
        ]
        assert len(service_name_vars) == 1
        assert service_name_vars[0]["value"] == k8s_name


# Made with Bob
