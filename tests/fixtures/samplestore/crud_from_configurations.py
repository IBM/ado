# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
from collections.abc import Callable

import pytest
from testcontainers.community.mysql import MySqlContainer

import ado.core.actuatorconfiguration.config
import ado.core.actuatorconfiguration.resource
import ado.core.discoveryspace.config
import ado.core.discoveryspace.space
import ado.core.resources
import ado.core.samplestore.config
import ado.metastore.project
from ado.core import ActuatorConfigurationResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.base import ActiveSampleStore
from ado.core.samplestore.config import SampleStoreConfiguration
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore


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

        from ado.core.samplestore.utils import create_sample_store_resource

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
    [ado.core.discoveryspace.config.DiscoverySpaceConfiguration, str],
    DiscoverySpace,
]:
    # Factory as fixture
    # ref: https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures
    def _create_space(
        configuration: ado.core.discoveryspace.config.DiscoverySpaceConfiguration,
        sample_store_id: str,
    ) -> DiscoverySpace:

        # We need to inject into the space configuration the sample store identifier
        configuration.sampleStoreIdentifier = sample_store_id

        space = ado.core.discoveryspace.space.DiscoverySpace.from_configuration(
            configuration,
            project_context=valid_ado_project_context,
            identifier=None,
        )

        space.saveSpace()
        return space

    return _create_space


@pytest.fixture
def create_actuatorconfiguration(
    sql_store: SQLStore,
) -> Callable[
    [ado.core.actuatorconfiguration.config.ActuatorConfiguration],
    ActuatorConfigurationResource,
]:
    def _create_actuatorconfiguration(
        configuration: ado.core.actuatorconfiguration.config.ActuatorConfiguration,
    ) -> ActuatorConfigurationResource:

        actuatorconfig_resource = ActuatorConfigurationResource(config=configuration)

        sql_store.addResource(actuatorconfig_resource)

        return actuatorconfig_resource

    return _create_actuatorconfiguration
