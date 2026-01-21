# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from collections.abc import Callable

import pytest
from testcontainers.mysql import MySqlContainer

import orchestrator.core.actuatorconfiguration.config
import orchestrator.core.actuatorconfiguration.resource
import orchestrator.core.discoveryspace.config
import orchestrator.core.discoveryspace.space
import orchestrator.core.resources
import orchestrator.core.samplestore.config
import orchestrator.metastore.project
from orchestrator.core import ActuatorConfigurationResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.samplestore.base import ActiveSampleStore
from orchestrator.core.samplestore.config import SampleStoreConfiguration
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore


@pytest.fixture
def create_sample_store(
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
) -> Callable[[SampleStoreConfiguration], ActiveSampleStore]:
    # Factory as fixture
    # ref: https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures
    def _create_sample_store(
        configuration: SampleStoreConfiguration,
    ) -> ActiveSampleStore:

        from orchestrator.core.samplestore.utils import create_sample_store_resource

        # To avoid having to provide passwords in the configuration
        # we need to inject them just like we do in ado create
        configuration.specification.storageLocation = (
            valid_ado_project_context.metadataStore
        )

        _, sample_store = create_sample_store_resource(
            configuration,
            sql_store,
        )

        return sample_store

    return _create_sample_store


@pytest.fixture
def create_space(
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
) -> Callable[
    [orchestrator.core.discoveryspace.config.DiscoverySpaceConfiguration, str],
    DiscoverySpace,
]:
    # Factory as fixture
    # ref: https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures
    def _create_space(
        configuration: orchestrator.core.discoveryspace.config.DiscoverySpaceConfiguration,
        sample_store_id: str,
    ) -> DiscoverySpace:

        # We need to inject into the space configuration the sample store identifier
        configuration.sampleStoreIdentifier = sample_store_id

        space = (
            orchestrator.core.discoveryspace.space.DiscoverySpace.from_configuration(
                configuration,
                project_context=valid_ado_project_context,
                identifier=None,
            )
        )

        space.saveSpace()
        return space

    return _create_space


@pytest.fixture
def create_actuatorconfiguration(
    sql_store: SQLStore,
) -> Callable[
    [orchestrator.core.actuatorconfiguration.config.ActuatorConfiguration],
    ActuatorConfigurationResource,
]:
    def _create_actuatorconfiguration(
        configuration: orchestrator.core.actuatorconfiguration.config.ActuatorConfiguration,
    ) -> ActuatorConfigurationResource:

        actuatorconfig_resource = ActuatorConfigurationResource(config=configuration)

        sql_store.addResource(actuatorconfig_resource)

        return actuatorconfig_resource

    return _create_actuatorconfiguration
