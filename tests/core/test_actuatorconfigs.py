# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import re

import pytest
import yaml

from orchestrator.core import ActuatorConfigurationResource
from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    OperatorReference,
)
from orchestrator.core.operation.resource import OperationResource
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.utilities.pydantic import ignore_plugin_validation_context


def test_nonexistent_actuatorconfig_raises_error() -> None:
    configuration = "tests/resources/nonexistent_actuatorconfiguration.yaml"

    with pytest.raises(
        ValueError,
        match=re.escape("Actuator nonexistent is not available in the registry"),
    ):
        ActuatorConfiguration.model_validate(
            yaml.safe_load(pathlib.Path(configuration).read_text())
        )


def test_actuatorconfiguration_ignores_plugin_validation_with_context() -> None:
    """Plugin validation can be skipped via validation context."""
    config = ActuatorConfiguration.model_validate(
        {"actuatorIdentifier": "nonexistent", "parameters": {}},
        context=ignore_plugin_validation_context,
    )
    assert config.actuatorIdentifier == "nonexistent"


def test_actuatorconfiguration_get_resource_without_plugin_validation(
    sql_store: SQLStore,
) -> None:
    """Stored actuator configs can be read without plugin validation on get."""
    resource = ActuatorConfigurationResource.model_validate(
        {
            "identifier": "actuatorconfiguration-nonexistent-test0001",
            "config": {"actuatorIdentifier": "nonexistent", "parameters": {}},
        },
        context=ignore_plugin_validation_context,
    )
    sql_store.addResource(resource)

    loaded = sql_store.getResource(
        identifier=resource.identifier,
        kind=CoreResourceKinds.ACTUATORCONFIGURATION,
    )
    assert loaded.config.actuatorIdentifier == "nonexistent"


def test_operation_get_resource_without_plugin_validation(
    sql_store: SQLStore,
) -> None:
    """Stored operations can be read without operator plugin validation on get."""
    operation_configuration = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        ),
        context=ignore_plugin_validation_context,
    )

    resource = OperationResource(
        operationType=DiscoveryOperationEnum.SEARCH,
        operatorIdentifier="randomwalk-0.1.0",
        config=operation_configuration,
    )
    sql_store.addResource(resource)

    loaded = sql_store.getResource(
        identifier=resource.identifier,
        kind=CoreResourceKinds.OPERATION,
    )
    assert loaded.config.operation.module.operatorName == "random_walk"


def test_operation_structural_parse_without_operator_validation() -> None:
    """Metastore read path: operation config parses without validating operator params."""
    operation_configuration = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        ),
        context=ignore_plugin_validation_context,
    )
    assert isinstance(operation_configuration.operation.module, OperatorReference)


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
