# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Unit tests for OTEL traces endpoint feature in vllm_performance actuator.
Tests parameter validation, YAML generation, and backward compatibility.
"""

import yaml

from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from plugins.actuators.vllm_performance.ado_actuators.vllm_performance.actuator_parameters import (
    VLLMPerformanceTestParameters,
)
from plugins.actuators.vllm_performance.ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
)


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
        assert params.otlp_traces_endpoint == url


class TestActuatorConfigurationWithOTLP:
    """Test suite for ActuatorConfiguration with otlp_traces_endpoint"""

    def test_actuator_configuration_with_otlp_endpoint(self) -> None:
        """Test full actuator configuration with OTLP endpoint"""
        config_yaml = """
actuatorIdentifier: vllm_performance
parameters:
  namespace: test-namespace
  otlp_traces_endpoint: http://jaeger:4318/v1/traces
"""
        config = ActuatorConfiguration(**yaml.safe_load(config_yaml))
        assert config.parameters.otlp_traces_endpoint == "http://jaeger:4318/v1/traces"  # type: ignore[union-attr]

    def test_actuator_configuration_without_otlp_endpoint(self) -> None:
        """Test actuator configuration without OTLP endpoint (backward compatibility)"""
        config_yaml = """
actuatorIdentifier: vllm_performance
parameters:
  namespace: test-namespace
  max_environments: 3
"""
        config = ActuatorConfiguration(**yaml.safe_load(config_yaml))
        assert config.parameters.otlp_traces_endpoint is None  # type: ignore[union-attr]

    def test_actuator_configuration_yaml_roundtrip(self) -> None:
        """Test YAML serialization roundtrip with OTLP endpoint"""
        config_yaml = """
actuatorIdentifier: vllm_performance
parameters:
  namespace: test-namespace
  otlp_traces_endpoint: http://jaeger:4318/v1/traces
  max_environments: 2
"""
        config = ActuatorConfiguration(**yaml.safe_load(config_yaml))

        # Serialize back to dict
        config_dict = config.model_dump()
        assert (
            config_dict["parameters"]["otlp_traces_endpoint"]
            == "http://jaeger:4318/v1/traces"
        )

        # Create new config from serialized dict
        config_restored = ActuatorConfiguration(**config_dict)
        assert config_restored.parameters.otlp_traces_endpoint == "http://jaeger:4318/v1/traces"  # type: ignore[union-attr]


class TestDeploymentYAMLWithOTLP:
    """Test suite for deployment YAML generation with otlp_traces_endpoint"""

    def test_deployment_yaml_without_otlp(self) -> None:
        """Test deployment YAML generation without OTLP endpoint"""
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            otlp_traces_endpoint=None,
        )

        # Verify no OTEL env var
        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env") or []
        otel_env = [
            e for e in env_vars if e["name"] == "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        ]
        assert len(otel_env) == 0

        # Verify no OTEL arg
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

        # Verify OTLP env var is present
        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env") or []
        otlp_env = [
            e for e in env_vars if e["name"] == "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        ]
        assert len(otlp_env) == 1
        assert otlp_env[0]["value"] == otlp_url

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

        # Verify it's the actual URL, not "$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        assert args[otlp_arg_index + 1] == otlp_url
        assert args[otlp_arg_index + 1] != "$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"


# Made with Bob
