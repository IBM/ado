# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import pytest
import yaml

import orchestrator.core
import orchestrator.core.operation.resource
from orchestrator.core import ADOResource
from orchestrator.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from orchestrator.core.operation.resource import OperationResource
from orchestrator.metastore.sqlstore import SQLStore


@pytest.fixture
def ml_multi_cloud_operation_resource(
    random_identifier: Callable[[], str],
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
) -> Callable[[str | None], OperationResource]:

    def _ml_multi_cloud_operation_resource(
        space_id: str | None = None,
    ) -> OperationResource:

        if space_id:
            ml_multi_cloud_operation_configuration.spaces = [space_id]

        return OperationResource(
            operationType=DiscoveryOperationEnum.SEARCH,
            operatorIdentifier=random_identifier(),
            config=ml_multi_cloud_operation_configuration,
        )

    return _ml_multi_cloud_operation_resource


@pytest.fixture
def ml_multi_cloud_operation_resource_from_db(
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    create_resources: Callable[[list[ADOResource], SQLStore], None],
) -> Callable[[str | None], OperationResource]:
    def _ml_multi_cloud_operation_resource_from_db(
        space_id: str | None = None,
    ) -> OperationResource:
        operation = ml_multi_cloud_operation_resource(space_id=space_id)
        create_resources(resources=[operation])
        return operation

    return _ml_multi_cloud_operation_resource_from_db


valid_operation_configs = [
    "tests/resources/operation/operation_config2.yaml",
    "tests/resources/operation/operation_config5.yaml",
    "tests/resources/operation/operation_config6b.yaml",
    "examples/pfas-generative-models/operation_random_walk_test.yaml",
    "examples/pfas-generative-models/operation_transformer_benchmark.yaml",
    "examples/optimization_test_functions/operation_nevergrad.yaml",
    "examples/optimization_test_functions/operation_bayesopt.yaml",
    "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml",
    "examples/ml-multi-cloud/raytune_ml_multicloud_operation.yaml",
    "examples/ml-multi-cloud/raytune_ml_multicloud_operation_custom_metric.yaml",
    "examples/ml-multi-cloud/lhc_sampler.yaml",
]


@pytest.fixture(params=valid_operation_configs)
def valid_operation_config_file(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def operation_configuration(
    test_space_identifier: str,
) -> DiscoveryOperationResourceConfiguration:

    # Return the default
    return DiscoveryOperationResourceConfiguration(
        spaces=[test_space_identifier], operation=DiscoveryOperationConfiguration()
    )


@pytest.fixture
def operation_resource(
    operation_configuration: DiscoveryOperationResourceConfiguration,
    test_space_identifier: str,
) -> OperationResource:

    operation_configuration.spaces = [test_space_identifier]

    # Create a random operation resource
    return OperationResource(
        config=operation_configuration,
        operationType=orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH,
        operatorIdentifier="randomwalk-0.3.1",
    )


@pytest.fixture
def test_operation_identifier(operation_resource: OperationResource) -> str:

    return operation_resource.identifier


@pytest.fixture
def random_walk_multicloud_operation_configuration() -> (
    DiscoveryOperationResourceConfiguration
):

    with open("examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml") as f:
        conf = DiscoveryOperationResourceConfiguration.model_validate(yaml.safe_load(f))

    # Remove values for the spaces
    conf.spaces = []
    return conf
