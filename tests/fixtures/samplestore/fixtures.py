# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from collections.abc import Callable

import pytest
from testcontainers.mysql import MySqlContainer

from orchestrator.core import (
    DiscoverySpaceResource,
    OperationResource,
    SampleStoreResource,
)
from orchestrator.core.samplestore.base import ActiveSampleStore
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore


@pytest.fixture
def sql_store(
    mysql_test_instance: MySqlContainer, valid_ado_project_context: ProjectContext
) -> SQLStore:
    return SQLStore(project_context=valid_ado_project_context)


@pytest.fixture
def sql_store_with_resources_preloaded(
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    discovery_space_resource: DiscoverySpaceResource,
    operation_resource: OperationResource,
    sample_store_resource: SampleStoreResource,
) -> SQLStore:

    # AP: 07/08/2025
    # This fixture is shared across a function invocation
    # as it is standard for Pytest.
    # see:https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
    #
    # Since requesting tests might have multiple parameters, this fixture
    # may be called multiple times. Attempting to add the same resources
    # multiple times, however, will raise exceptions, so we need to check
    # whether these resources exist before we try to add them

    sql = SQLStore(project_context=valid_ado_project_context)

    # Ensure our resources are consistent
    if not sql.containsResourceWithIdentifier(
        identifier=sample_store_resource.identifier
    ):
        sql.addResource(sample_store_resource)

    # The space must belong to the sample store
    if not sql.containsResourceWithIdentifier(discovery_space_resource.identifier):
        discovery_space_resource.config.sampleStoreIdentifier = (
            sample_store_resource.identifier
        )
        sql.addResourceWithRelationships(
            resource=discovery_space_resource,
            relatedIdentifiers=[sample_store_resource.identifier],
        )

    # The operation must belong to the space
    if not sql.containsResourceWithIdentifier(operation_resource.identifier):
        operation_resource.config.spaces = [discovery_space_resource.identifier]
        sql.addResourceWithRelationships(
            resource=operation_resource,
            relatedIdentifiers=operation_resource.config.spaces,
        )

    return sql


@pytest.fixture
def empty_sample_store(
    create_sample_store: Callable[[SampleStoreConfiguration], ActiveSampleStore],
) -> SQLSampleStore:
    sample_store_configuration = SampleStoreConfiguration(
        specification=SampleStoreSpecification(
            module=SampleStoreModuleConf(
                moduleClass="SQLSampleStore",
                moduleName="orchestrator.core.samplestore.sql",
            ),
        )
    )
    return create_sample_store(sample_store_configuration)
