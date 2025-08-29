# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pytest

import orchestrator.core.actuatorconfiguration.config
import orchestrator.core.actuatorconfiguration.resource
import orchestrator.core.discoveryspace.config
import orchestrator.core.discoveryspace.space
import orchestrator.core.resources
import orchestrator.core.samplestore.config
import orchestrator.metastore.project


@pytest.fixture
def create_sample_store(sql_store, valid_ado_project_context):
    # Factory as fixture
    # ref: https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures
    def _create_sample_store(
        configuration: orchestrator.core.samplestore.config.SampleStoreConfiguration,
    ):

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
    mysql_test_instance,
    valid_ado_project_context,
):
    # Factory as fixture
    # ref: https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures
    def _create_space(
        configuration: orchestrator.core.discoveryspace.config.DiscoverySpaceConfiguration,
        sample_store_id: str,
    ):

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
    sql_store,
):
    def _create_actuatorconfiguration(
        configuration: orchestrator.core.actuatorconfiguration.config.ActuatorConfiguration,
    ):
        import orchestrator.metastore.sqlstore

        actuatorconfig_resource = orchestrator.core.actuatorconfiguration.resource.ActuatorConfigurationResource(
            config=configuration
        )

        sql_store.addResource(actuatorconfig_resource)

        return actuatorconfig_resource

    return _create_actuatorconfiguration
