# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
import re

import pydantic
import pytest
import yaml

from orchestrator.core import ActuatorConfigurationResource
from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationResourceConfiguration,
)
from orchestrator.metastore.project import ProjectContext


def test_nonexistent_actuatorconfig_raises_error() -> None:
    configuration = "tests/resources/nonexistent_actuatorconfiguration.yaml"

    with pytest.raises(
        pydantic.ValidationError,
        match=re.escape("Actuator nonexistent is not available in the registry"),
    ):
        ActuatorConfiguration.model_validate(
            yaml.safe_load(pathlib.Path(configuration).read_text())
        )


def test_ml_multi_cloud_operation_valid(
    valid_ado_project_context: ProjectContext,
    ml_multi_cloud_correct_actuatorconfiguration: ActuatorConfigurationResource,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:

    operation_configuration = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )

    # Overrides
    operation_configuration.spaces = [ml_multi_cloud_space.uri]
    operation_configuration.actuatorConfigurationIdentifiers = [
        ml_multi_cloud_correct_actuatorconfiguration.identifier
    ]

    operation_configuration.validate_actuatorconfigurations(
        project_context=valid_ado_project_context
    )


def test_ml_multi_cloud_operation_invalid(
    valid_ado_project_context: ProjectContext,
    ml_multi_cloud_invalid_actuatorconfiguration: ActuatorConfigurationResource,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:

    operation_configuration = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )

    # Overrides
    operation_configuration.spaces = [ml_multi_cloud_space.uri]
    operation_configuration.actuatorConfigurationIdentifiers = [
        ml_multi_cloud_invalid_actuatorconfiguration.identifier
    ]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Actuator Identifiers {'mock'} must appear in the experiments of its space"
        ),
    ):
        operation_configuration.validate_actuatorconfigurations(
            project_context=valid_ado_project_context
        )


def test_ml_multi_cloud_operation_base_get(
    valid_ado_project_context: ProjectContext,
    ml_multi_cloud_correct_actuatorconfiguration: ActuatorConfigurationResource,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Tests directly that BaseOperationRunConfiguration works"""
    operation_configuration = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )

    operation_configuration.get_actuatorconfigurations(
        project_context=valid_ado_project_context
    )
