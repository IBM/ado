# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

import orchestrator.core
import orchestrator.metastore.sqlstore
from orchestrator.core.samplestore.base import SampleStore
from orchestrator.core.samplestore.config import SampleStoreConfiguration
from orchestrator.core.samplestore.resource import SampleStoreResource

moduleLogger = logging.getLogger("sample-store-utils")


def create_sample_store_resource(
    configuration: SampleStoreConfiguration,
    resource_store: orchestrator.metastore.sqlstore.SQLStore,
) -> tuple[
    SampleStoreResource,
    orchestrator.core.samplestore.base.ActiveSampleStore,
]:
    """Creates a SampleStore based on a configuration and stores it in the resource store.

    The SampleStore must be an active sample store.

    Args:
        configuration: Configuration for the SampleStore to create
        resource_store: The SQLStore to persist the resource in

    Returns:
        Tuple of (SampleStoreResource, ActiveSampleStore instance)
    """

    source = SampleStore.from_configuration(configuration)

    # Create and store resource
    resource = SampleStoreResource(identifier=source.identifier, config=configuration)

    # Note: The resource store will apply custom dump/load for SQLSampleStores
    # This removes/re-adds the storage location info
    resource_store.addResource(resource=resource)

    return resource, source
