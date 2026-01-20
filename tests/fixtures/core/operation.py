# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import pathlib
import uuid
from collections.abc import Callable

import pytest
import yaml

import orchestrator.core
import orchestrator.core.operation.resource
from orchestrator.core import ADOResource
from orchestrator.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationResourceConfiguration,
)
from orchestrator.core.operation.resource import OperationResource
from orchestrator.metastore.sqlstore import SQLStore


@pytest.fixture
def random_operation_resource_from_file(
    random_identifier: Callable[[], str],
) -> Callable[[str | None, str | None], OperationResource]:

    def _random_operation_resource_from_file(
        sample_store_id: str | None = None,
        space_id: str | None = None,
    ) -> OperationResource:
        file = pathlib.Path("tests/resources/operation/operation_resource.json")
        random_resource_id = str(uuid.uuid4())
        if not sample_store_id:
            sample_store_id = random_identifier()
        if not space_id:
            space_id = random_identifier()

        # Get the model
        operation = (
            orchestrator.core.operation.resource.OperationResource.model_validate(
                json.loads(file.read_text())
            )
        )

        # Final touch-ups
        operation.identifier = random_resource_id
        operation.config.spaces = [space_id]
        return operation

    return _random_operation_resource_from_file


@pytest.fixture
def random_operation_resource_from_db(
    random_operation_resource_from_file: Callable[
        [str | None, str | None], OperationResource
    ],
    create_resources: Callable[[list[ADOResource], SQLStore], None],
) -> Callable[[str | None, str | None], OperationResource]:
    def _random_operation_resource_from_db(
        sample_store_id: str | None = None,
        space_id: str | None = None,
    ) -> OperationResource:
        operation = random_operation_resource_from_file(
            sample_store_id=sample_store_id, space_id=space_id
        )
        create_resources(resources=[operation])
        return operation

    return _random_operation_resource_from_db


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
