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


class TestOTELTracesEndpointParameter:
    """Test suite for otel_traces_endpoint parameter in VLLMPerformanceTestParameters"""

    def test_otel_traces_endpoint_optional(self) -> None:
        """Test that otel_traces_endpoint is optional and defaults to None"""
        params = VLLMPerformanceTestParameters()  # type: ignore[call-arg]
        assert params.otel_traces_endpoint is None

    def test_otel_traces_endpoint_accepts_valid_url(self) -> None:
        """Test that otel_traces_endpoint accepts valid URLs"""
        url = "http://jaeger:4318/v1/traces"
        params = VLLMPerformanceTestParameters(otel_traces_endpoint=url)  # type: ignore[call-arg]
        assert params.otel_traces_endpoint == url


class TestActuatorConfigurationWithOTEL:
    """Test suite for ActuatorConfiguration with otel_traces_endpoint"""

    def test_actuator_configuration_with_otel_endpoint(self) -> None:
        """Test full actuator configuration with OTEL endpoint"""
        config_yaml = """
actuatorIdentifier: vllm_performance
parameters:
  namespace: test-namespace
  otel_traces_endpoint: http://jaeger:4318/v1/traces
"""
        config = ActuatorConfiguration(**yaml.safe_load(config_yaml))
        assert config.parameters.otel_traces_endpoint == "http://jaeger:4318/v1/traces"  # type: ignore[union-attr]

    def test_actuator_configuration_without_otel_endpoint(self) -> None:
        """Test actuator configuration without OTEL endpoint (backward compatibility)"""
        config_yaml = """
actuatorIdentifier: vllm_performance
parameters:
  namespace: test-namespace
  max_environments: 3
"""
        config = ActuatorConfiguration(**yaml.safe_load(config_yaml))
        assert config.parameters.otel_traces_endpoint is None  # type: ignore[union-attr]

    def test_actuator_configuration_yaml_roundtrip(self) -> None:
        """Test YAML serialization roundtrip with OTEL endpoint"""
        config_yaml = """
actuatorIdentifier: vllm_performance
parameters:
  namespace: test-namespace
  otel_traces_endpoint: http://jaeger:4318/v1/traces
  max_environments: 2
"""
        config = ActuatorConfiguration(**yaml.safe_load(config_yaml))

        # Serialize back to dict
        config_dict = config.model_dump()
        assert (
            config_dict["parameters"]["otel_traces_endpoint"]
            == "http://jaeger:4318/v1/traces"
        )

        # Create new config from serialized dict
        config_restored = ActuatorConfiguration(**config_dict)
        assert config_restored.parameters.otel_traces_endpoint == "http://jaeger:4318/v1/traces"  # type: ignore[union-attr]


class TestDeploymentYAMLWithOTEL:
    """Test suite for deployment YAML generation with otel_traces_endpoint"""

    def test_deployment_yaml_without_otel(self) -> None:
        """Test deployment YAML generation without OTEL endpoint"""
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            otel_traces_endpoint=None,
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

    def test_deployment_yaml_with_otel(self) -> None:
        """Test deployment YAML generation with OTEL endpoint"""
        otel_url = "http://jaeger:4318/v1/traces"
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            otel_traces_endpoint=otel_url,
        )

        # Verify OTEL env var is present
        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env") or []
        otel_env = [
            e for e in env_vars if e["name"] == "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        ]
        assert len(otel_env) == 1
        assert otel_env[0]["value"] == otel_url

        # Verify OTEL arg is present with correct value
        args = container["args"]
        assert "--otlp-traces-endpoint" in args
        otel_arg_index = args.index("--otlp-traces-endpoint")
        assert args[otel_arg_index + 1] == otel_url

    def test_deployment_yaml_otel_arg_not_env_var_reference(self) -> None:
        """Test that OTEL arg uses actual value, not environment variable reference"""
        otel_url = "http://jaeger:4318/v1/traces"
        yaml_dict = ComponentsYaml.deployment_yaml(
            k8s_name="test-deployment",
            model="test-model",
            otel_traces_endpoint=otel_url,
        )

        container = yaml_dict["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]
        otel_arg_index = args.index("--otlp-traces-endpoint")

        # Verify it's the actual URL, not "$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        assert args[otel_arg_index + 1] == otel_url
        assert args[otel_arg_index + 1] != "$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"


# Made with Bob
