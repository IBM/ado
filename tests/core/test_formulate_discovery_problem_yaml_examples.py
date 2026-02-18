# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests that formulate-discovery-problem YAML examples parse via ado pydantic models."""

import pathlib

import pytest
import yaml

from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.operation.config import DiscoveryOperationResourceConfiguration

YAML_EXAMPLES_DIR = pathlib.Path(
    ".cursor/skills/formulate-discovery-problem/yaml-examples"
)

# Space YAML files: DiscoverySpaceConfiguration structure
# (sampleStoreIdentifier, entitySpace, experiments, metadata)
SPACE_YAML_PATTERNS = [
    "example1-space",
    "example2-space",
    "example3-space",
    "example4-space",
    "example5-space",
    "reference-pattern1-simple-space",
    "reference-pattern2-multiple-experiments",
    "reference-pattern3-parameterized",
    "reference-pattern4-optional-property",
    "reference-experiment-format",
    "skill-manual-structure",
    "error-fix-batch-size",
]

# Operation YAML files: DiscoveryOperationResourceConfiguration structure
# Uses random_walk operator (available in test env) or structural examples
OPERATION_YAML_PATTERNS = [
    "example1-operation",
    "example6-operation",
    "skill-operation-structure",
]

# Actuator config YAML files: ActuatorConfiguration structure
# Uses mock actuator (available in test env)
ACTUATOR_CONFIG_YAML_PATTERNS = [
    "skill-actuator-config-mock",
]


@pytest.mark.parametrize(
    "yaml_name",
    SPACE_YAML_PATTERNS,
    ids=SPACE_YAML_PATTERNS,
)
def test_agent_space_yaml_examples(yaml_name: str) -> None:
    """Each formulate-discovery-problem space YAML parses as DiscoverySpaceConfiguration."""
    base = pathlib.Path(__file__).resolve().parent.parent.parent / YAML_EXAMPLES_DIR
    path = base / f"{yaml_name}.yaml"
    if not path.exists():
        pytest.skip(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text())
    config = DiscoverySpaceConfiguration.model_validate(data)
    assert config is not None


@pytest.mark.parametrize(
    "yaml_name",
    OPERATION_YAML_PATTERNS,
    ids=OPERATION_YAML_PATTERNS,
)
def test_agent_operation_yaml_examples(yaml_name: str) -> None:
    """Each formulate-discovery-problem operation YAML parses as DiscoveryOperationResourceConfiguration."""
    base = pathlib.Path(__file__).resolve().parent.parent.parent / YAML_EXAMPLES_DIR
    path = base / f"{yaml_name}.yaml"
    if not path.exists():
        pytest.skip(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text())
    config = DiscoveryOperationResourceConfiguration.model_validate(data)
    assert config is not None


@pytest.mark.parametrize(
    "yaml_name",
    ACTUATOR_CONFIG_YAML_PATTERNS,
    ids=ACTUATOR_CONFIG_YAML_PATTERNS,
)
def test_agent_actuator_config_yaml_examples(yaml_name: str) -> None:
    """Each formulate-discovery-problem actuator config YAML parses as ActuatorConfiguration."""
    base = pathlib.Path(__file__).resolve().parent.parent.parent / YAML_EXAMPLES_DIR
    path = base / f"{yaml_name}.yaml"
    if not path.exists():
        pytest.skip(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text())
    config = ActuatorConfiguration.model_validate(data)
    assert config is not None
