# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

import orchestrator.core
import orchestrator.metastore.sqlstore
import orchestrator.modules.module
import orchestrator.utilities.location
from orchestrator.core.samplestore.base import SampleStore
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreSpecification,
)
from orchestrator.core.samplestore.resource import SampleStoreResource

moduleLogger = logging.getLogger("sample-store-utils")


def initialize_sample_store_from_specification(
    identifier: str | None,
    spec: SampleStoreSpecification,
) -> SampleStore:
    """Load an existing SampleStore called identifier, or creates a new one if no identifier given

    Parameters:
        identifier: The identifier of the SampleStore to initialize. If None a new SampleStore will be created
        spec: The specification of the SampleStore. Tells where it is, how to access it, and what class to use.
    """

    sourceClass = orchestrator.modules.module.load_module_class_or_function(spec.module)
    moduleLogger.debug(f"Class: {sourceClass}, Params: {spec.parameters}")

    # Convert dict parameters back to model object if needed
    # (parameters are stored as dict for serialization, but constructors expect model objects)
    if isinstance(spec.parameters, dict):
        # Always use validate_parameters - it handles both raw and pre-validated dicts
        parameters = sourceClass.validate_parameters(parameters=spec.parameters)
    else:
        parameters = spec.parameters

    return sourceClass(
        identifier=identifier,
        storageLocation=spec.storageLocation,
        parameters=parameters,
    )


def initialize_sample_store_from_reference(
    reference: orchestrator.core.samplestore.config.SampleStoreReference,
) -> SampleStore:
    """Load an existing SampleStore using reference

    Parameters:
        reference: A reference to the SampleStore. Tells where it is, how to access it, and what class to use.
    """

    sourceClass = orchestrator.modules.module.load_module_class_or_function(
        reference.module
    )
    moduleLogger.debug(f"Class: {sourceClass}, Params: {reference.parameters}")

    # Convert dict parameters back to model object if needed
    # (parameters are stored as dict for serialization, but constructors expect model objects)
    if isinstance(reference.parameters, dict):
        # Always use validate_parameters - it handles both raw and pre-validated dicts
        parameters = sourceClass.validate_parameters(parameters=reference.parameters)
    else:
        parameters = reference.parameters

    if reference.identifier is not None:
        # For sample stores that requires a separate identifier
        # because their storageLocation is a container for possibly many sample stores
        return sourceClass(
            identifier=reference.identifier,
            storageLocation=reference.storageLocation,
            parameters=parameters,
        )
    # for sample stores that are equivalent to the storageLocation i.e. identifier is part of storageLocation
    return sourceClass(
        storageLocation=reference.storageLocation,
        parameters=parameters,
    )


def load_sample_store_from_resource(
    sample_store_resource: SampleStoreResource,
) -> SampleStore:
    """Returns a SampleStore instance for the SampleStore described by sample_store_resource

    Params:
        sample_store_resource: An SampleStoreResource describing an exiting SampleStore
    """

    return initialize_sample_store_from_specification(
        identifier=sample_store_resource.identifier,
        spec=sample_store_resource.config.specification,
    )


def create_sample_store(
    conf: SampleStoreConfiguration,
) -> orchestrator.core.samplestore.base.ActiveSampleStore:
    """Creates a SampleStore based on a configuration"""

    if conf.specification is None:
        raise ValueError("Your sample store does not contain the specification field")

    sample_store = initialize_sample_store_from_specification(
        identifier=None, spec=conf.specification
    )

    #
    # Copy in data
    #
    additional_sample_stores = []
    if conf.copyFrom is not None:
        additional_sample_stores = [
            initialize_sample_store_from_reference(reference)
            for reference in conf.copyFrom
        ]

    for s in additional_sample_stores:  # type: SampleStore
        moduleLogger.debug(
            f"Copying {s.numberOfEntities} entities from {s.identifier} to {sample_store.identifier}"
        )
        sample_store.add_external_entities(s.entities)

    return sample_store


def create_sample_store_resource(
    conf: SampleStoreConfiguration,
    resourceStore: orchestrator.metastore.sqlstore.SQLStore,
) -> tuple[
    SampleStoreResource,
    orchestrator.core.samplestore.base.ActiveSampleStore,
]:
    """Creates a SampleStore based on a configuration and stores it in the resource store

    The SampleStore must be an active sample store
    """

    source = create_sample_store(conf)

    #
    # Create and store resource
    #

    resource = SampleStoreResource(identifier=source.identifier, config=conf)

    # MJ: Note, the resource store will apply custom dump/load for SQLSampleStores
    # This removes/re-adds the storage location info
    resourceStore.addResource(resource=resource)

    return resource, source
