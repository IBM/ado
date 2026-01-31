# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import typing
from collections.abc import Callable
from functools import wraps
from typing import Any

import orchestrator.core.discoveryspace.resource
import orchestrator.core.metadata
import orchestrator.core.resources
import orchestrator.core.samplestore.base
import orchestrator.schema.entity
import orchestrator.schema.measurementspace
import orchestrator.schema.property_value
import orchestrator.schema.virtual_property
import orchestrator.utilities.logging
from orchestrator.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
    DiscoverySpaceProperties,
)
from orchestrator.core.operation.resource import OperationResource
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.actuators.catalog import ActuatorCatalogExtension
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.entity import Entity
from orchestrator.schema.entityspace import (
    EntitySpaceRepresentation,
)
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.property_value import constitutive_property_values_from_point
from orchestrator.schema.request import MeasurementRequest
from orchestrator.schema.result import MeasurementResult

if typing.TYPE_CHECKING:
    from pandas import DataFrame
    from rich.console import RenderableType

    from orchestrator.metastore.sqlstore import SQLStore

FORMAT = orchestrator.utilities.logging.FORMAT
LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, format=FORMAT)

moduleLogger = logging.getLogger("discoveryspace")


def _perform_preflight_checks_for_sample_store_methods(
    f: Callable[..., Any],  # noqa: ANN401
) -> Callable[["DiscoverySpace", tuple[Any, ...], dict[str, Any]], Any]:  # noqa: ANN401
    """
    Performs common checks on DiscoverySpace methods that wrap
    SQLSampleStore methods.

    Checks include:
    - Ensuring the DiscoverySpace.sample_store is of type SQLSampleStore
    - Ensuring the operation_id passed as a parameter belongs to the DiscoverySpace
    """

    @wraps(f)
    def perform_checks(
        self: "DiscoverySpace", *args: Any, **kwargs: Any  # noqa: ANN401
    ) -> Any:  # noqa: ANN401

        import orchestrator.core.samplestore.sql

        if not isinstance(
            self.sample_store, orchestrator.core.samplestore.sql.SQLSampleStore
        ):
            raise ValueError(
                "The complete_measurement_request_with_results_timeseries method "
                "requires the use of an SQLSampleStore"
            )

        operation_id = kwargs.get("operation_id") or args[0]
        space_for_operation = self._metadataStore.getResource(
            identifier=operation_id,
            kind=CoreResourceKinds.OPERATION,
            raise_error_if_no_resource=True,
        ).config.spaces[0]

        if self.uri != space_for_operation:
            raise ValueError(
                f"Operation {operation_id} does not belong to space {self.uri}, but rather to {space_for_operation}"
            )

        return f(self, *args, **kwargs)

    return perform_checks


class SpaceInconsistencyError(Exception):
    """When the characteristics of the DiscoverySpace are found to be inconsistent with the DiscoverySpace definition

    For example, a space cannot contain entities with the same id"""


class DiscoverySpace:
    """
    Represents a DiscoverySpace

    A discovery space has the following required properties:

    sample_store: This is where to read entities from and store/update entities to
    measurementSpace: Describes the measurement space
    metadataStore: A place to store metadata about the space and metrics

    An DiscoverySpace which only has a SampleStore cannot generate new entities.
    It can only be used to operate on what is in the store.

    An DiscoverySpace can have three additional properties:

    entitySpace: This provides information required to generate new entities
    sampleGenerator: This provides a way to generate entities from the entitySpaceRepresentation
    sampleSelector: This provides a way to select entities from the sample_store

    Note: A sampleGenerator requires an entitySpace.
    Note: You can use other samplers to sample from the EntitySpace. The ones provided here will act as defaults.
    """

    PropertyFormatType = typing.Literal["observed", "target"]

    @classmethod
    def from_configuration(
        cls,
        conf: "DiscoverySpaceConfiguration",
        project_context: ProjectContext,
        identifier: str | None = None,
    ) -> "DiscoverySpace":
        """Creates a discovery space from a config

        Params:
            conf: A DiscoverySpaceConfiguration object that contains the information required to create the space
            project_context: A ProjectContext object that will enable the returned discovery space
                to retrieve and store information on operations on the space and related metrics in a remote db
            identifier: An optional identifier for the space. If None one will be generated by the DiscoverySpace
                Note: The identifier is required to find relevant data in storage.
                Thus, if conf is a stored conf you must also pass the stored identifier here.
                Otherwise, a new space not connected to the previous stored data may be created (depends on how the
                discovery space generates the id versus how the id used to store was generated)

        """

        from orchestrator.core.samplestore.utils import (
            load_sample_store_from_resource,
        )
        from orchestrator.metastore.sqlstore import SQLStore

        resourceStore = SQLStore(project_context=project_context)

        entitySpace = None

        resource = resourceStore.getResource(
            identifier=conf.sampleStoreIdentifier,
            kind=CoreResourceKinds.SAMPLESTORE,
            raise_error_if_no_resource=True,
        )
        sample_store = load_sample_store_from_resource(resource)

        if conf.entitySpace is not None:
            entitySpace = EntitySpaceRepresentation.representationFromConfiguration(
                conf.entitySpace
            )

        ## Add any external experiments to the replay actuators catalog
        externalCatalogs = []
        if sample_store is not None:
            moduleLogger.debug(
                f"Loading external experiments from sample store: {sample_store.identifier}"
            )

            catalog = sample_store.experimentCatalog()
            if catalog is not None:
                externalCatalogs.append(catalog)
                moduleLogger.debug(
                    f"Loaded external catalog {catalog} based on sample store {sample_store}"
                )
                ActuatorRegistry.globalRegistry().updateCatalogs(
                    ActuatorCatalogExtension(experiments=catalog.experiments)
                )
                moduleLogger.debug(
                    ActuatorRegistry.globalRegistry()
                    .catalogForActuatorIdentifier("replay")
                    .experiments
                )

        if isinstance(
            conf.experiments,
            orchestrator.schema.measurementspace.MeasurementSpaceConfiguration,
        ):
            # If we have full MeasurementSpaceConfiguration we can initialize directly
            measurementSpace = MeasurementSpace(configuration=conf.experiments)
        else:
            # Otherwise we have to use registry and additional catalogs to reconstruct the experiments
            measurementSpace = MeasurementSpace.measurementSpaceFromSelection(
                selectedExperiments=conf.experiments,
                experimentCatalogs=externalCatalogs,
            )

        return cls(
            identifier=identifier,
            sample_store=sample_store,
            entitySpace=entitySpace,
            measurementSpace=measurementSpace,
            project_context=project_context,
            metadata=conf.metadata,
        )

    @classmethod
    def from_stored_configuration(
        cls, project_context: ProjectContext, space_identifier: str
    ) -> "DiscoverySpace":

        from orchestrator.metastore.sqlstore import SQLStore

        moduleLogger.debug("Accessing discovery space metadata store")
        metadataStore = SQLStore(project_context=project_context)
        moduleLogger.debug(
            f"Retrieving configuration for discovery space {space_identifier}"
        )
        resource = metadataStore.getResource(
            identifier=space_identifier,
            kind=CoreResourceKinds.DISCOVERYSPACE,
            raise_error_if_no_resource=True,
        )
        conf = resource.config

        moduleLogger.debug(f"Retrieved configuration is: {conf}")

        # project_context will define connection to the metadata storage via a particular host
        # For example the MySQL db has been port forwarded and is accessible on `localhost`
        # In the conf returned from the database there will be a configuration for the sample_store
        # which may be the same database as the metadata store.
        # However, when the state was stored it may have been by a different route
        # For example the sample_store is in the db now on localhost but when stored the host was percona-mysql-haproxy
        # This route will be inaccessible, and then you will not be able to load the sample store

        moduleLogger.debug("Initialising discovery space using stored configuration")
        return cls.from_configuration(
            conf=conf,
            project_context=project_context,
            identifier=space_identifier,
        )

    @classmethod
    def from_operation_id(
        cls, operation_id: str, project_context: ProjectContext
    ) -> "DiscoverySpace":
        """
        Creates a DiscoverySpace instance of the class from the given operation id and project context.

        Args:
            operation_id (str): The operation id to be used for finding the space identifier.
            project_context (ProjectContext): The project context to be used for creating the discovery space.

        Returns:
            DiscoverySpace: The newly created discovery space instance.

        Raises:
            ResourceDoesNotExistError: If the specified operation or related space do not exist.
            NoRelatedResourcesError: If no sample store is associated with the specified operation or related space.
        """
        from orchestrator.metastore.sqlstore import SQLResourceStore

        sql = SQLResourceStore(project_context=project_context)

        # FIXME AP 12/06/2025:
        # We are using the first space - which may become a problem in the future
        space_id = sql.getResource(
            identifier=operation_id,
            kind=CoreResourceKinds.OPERATION,
            raise_error_if_no_resource=True,
        ).config.spaces[0]

        return cls.from_stored_configuration(
            project_context=project_context,
            space_identifier=space_id,
        )

    def __init__(
        self,
        project_context: ProjectContext,
        identifier: str | None = None,
        sample_store: (
            orchestrator.core.samplestore.base.ActiveSampleStore | None
        ) = None,
        entitySpace: EntitySpaceRepresentation | None = None,
        measurementSpace: MeasurementSpace | None = None,
        properties: (
            orchestrator.core.discoveryspace.config.DiscoverySpaceProperties | None
        ) = None,
        metadata: orchestrator.core.metadata.ConfigurationMetadata | None = None,
    ) -> None:
        """

        Parameters:

            identifier: The identifier of this space. If not specified the receiver will generate one
            sample_store: Where to read entities from and store them to
            entitySpace: A representation of the mathematical space the entities are
            from. Can be None if this is not None i.e. it is implicit in the currently sampled entities.
            measurementSpace:
            properties: A DiscoverySpaceProperties object containing information on the spaces characteristics
            project_context: Contains information for connecting to backend databases.
                If None the backends try to initialise themselves based on env-vars
                The project name will be "default"

        Raises:
            SpaceInconsistencyError if:
            1. The MeasurementSpace does not contain an experiment measuring an observed property
               required by another experiment in the space
            2. The EntitySpace is inconsistent with the measurement space
        """

        import uuid

        if not properties:
            properties = DiscoverySpaceProperties()

        self.log = logging.getLogger("discovery-space")

        if not measurementSpace.isConsistent:
            raise SpaceInconsistencyError(
                "MeasurementSpace does not contain an experiment measuring an observed property"
                " required by another experiment in the space "
            )

        if entitySpace and measurementSpace:
            try:
                measurementSpace.checkEntitySpaceCompatible(entitySpace)
            except ValueError as error:
                raise SpaceInconsistencyError(
                    f"The entity space is not compatible with the measurement space: {error}"
                ) from error

        self._sample_store = sample_store
        self._measurementSpace = measurementSpace
        self._entitySpace = entitySpace
        self._properties = properties
        self._metadata = metadata

        if project_context is None:
            raise ValueError(
                "DiscoverySpace requires a valid ProjectContext to be passed."
            )
        self.log.debug("Using supplied project context")
        self._project_context = project_context.model_copy(deep=True)

        self.log.debug(
            f"Project context for DiscoverySpace is: {self._project_context}"
        )

        # Access metadata store
        from orchestrator.metastore.sqlstore import SQLStore

        self._metadataStore = SQLStore(project_context=project_context)

        self._identifier = (
            identifier
            if identifier is not None
            else f"space-{str(uuid.uuid4())[:6]}-{self._sample_store.identifier}"
        )

    def __rich__(self) -> "RenderableType":
        """Rich console representation of the DiscoverySpace."""
        import rich.box
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        components = [
            Text.assemble(("Identifier: ", "bold"), (self.uri, "bold green")),
        ]

        if self.entitySpace is not None:
            components.extend(
                [
                    Text("Entity Space:", style="bold"),
                    Panel(self.entitySpace, box=rich.box.SIMPLE_HEAD),
                ]
            )

        # MeasurementSpace has __rich__() method
        components.extend(
            [
                Text("Measurement Space:", style="bold"),
                Panel(self.measurementSpace, box=rich.box.SIMPLE_HEAD),
            ]
        )

        components.extend(
            [
                Text("Sample Store:", style="bold"),
                Panel(self.sample_store, box=rich.box.SIMPLE_HEAD),
            ]
        )

        return Group(*components)

    @property
    def uri(self) -> str:
        """Return an identifier for the space"""

        return self._identifier

    @property
    def project_context(self) -> ProjectContext:
        """Returns information required to retrieve/recreate the receiver from a metadata store"""

        return self._project_context

    @property
    def measurementSpace(self) -> MeasurementSpace:

        return self._measurementSpace

    @property
    def sample_store(
        self,
    ) -> orchestrator.core.samplestore.base.ActiveSampleStore:
        """Returns the sample store"""

        return self._sample_store

    @property
    def entitySpace(self) -> EntitySpaceRepresentation:
        """Returns the sample store"""

        return self._entitySpace

    @property
    def properties(self) -> DiscoverySpaceProperties:

        return self._properties

    @property
    def config(self) -> DiscoverySpaceConfiguration:

        # FIXME: entitySpace definition in config is explicit entity space ...
        entitySpaceConf = None
        if self.entitySpace is not None:
            entitySpaceConf = self.entitySpace.config

        # Note: We store the selfContainedConfig (MeasurementSpaceConfig)
        # as this means the actuators/registry will not have to be queried to rebuild the measurement space
        # This is problematic as all actuators used by the space would have to be loaded requiring one or both of
        # (a) an explicit import of orchestrator.actuators.base (b) information on all dynamic actuator modules used.

        metadata = (
            self._metadata
            if self._metadata is not None
            else orchestrator.core.metadata.ConfigurationMetadata()
        )

        return DiscoverySpaceConfiguration(
            sampleStoreIdentifier=self.sample_store.identifier,
            entitySpace=entitySpaceConf,
            experiments=self.measurementSpace.selfContainedConfig,
            metadata=metadata,
        )

    @property
    def resource(
        self,
    ) -> orchestrator.core.discoveryspace.resource.DiscoverySpaceResource:

        return orchestrator.core.discoveryspace.resource.DiscoverySpaceResource(
            identifier=self._identifier, config=self.config
        )

    def saveSpace(self) -> None:
        """Record this space in the metadata store"""

        if self.metadataStore is not None:
            try:
                self.metadataStore.addResource(resource=self.resource)
            except ValueError:
                pass
            else:
                # Add a relationship between this object and the samplestore
                # Note the DiscoverySpace is the subject as it is dependent on the
                # SampleStore but not vice versa
                self.metadataStore.addRelationship(
                    self.sample_store.identifier, self.uri
                )
        else:
            self.log.warning(
                f"Unable to store space {self._identifier} as no metadata storage provided"
            )

    def sampledEntities(self) -> list[Entity]:
        """Returns the entities sampled so far in the space"""

        # find all sampled entities in this space
        sampled_entity_ids = []
        for operationid in self.operations["IDENTIFIER"]:
            sampled_entity_ids.extend(
                self.entity_identifiers_in_operation(operation_id=operationid)
            )

        sampled_entity_ids_set = set(sampled_entity_ids)

        # Get all entities in the store
        all_entities = self.sample_store.entities

        # TODO: Consider removing isEntitySpace check
        # The additional check of isEntityInSpace should not be required if things are working correctly
        # However if an entity was incorrectly sampled during an operation, due to a bug say, this will correct for it
        return [
            e
            for e in all_entities
            if e.identifier in sampled_entity_ids_set
            and self.entitySpace.isEntityInSpace(e)
        ]

    def matchingEntities(self) -> list[Entity]:
        """Returns all entities in the sample store that match the space

        Note: They do not have to have any measurements from the measurement space

        If
        - ExplicitEntitySpace defined -> filter on the space
        - No space defined -> implies the space entities == all entities in the source.
        """

        # Get all entities in the store
        all_entities = self.sample_store.entities
        if self.entitySpace is not None:
            entities = [e for e in all_entities if self.entitySpace.isEntityInSpace(e)]
        else:
            entities = all_entities

        return entities

    def addMeasurement(self, request: MeasurementRequest) -> None:
        """Adds a measurement on an entity to the space

        Params:
            request: A MeasurementRequest object representing the measurement of properties of entities
                by a specific experiment
        """

        self.sample_store.addMeasurement(measurementRequest=request)

    def measuredEntitiesTable(
        self,
        property_type: PropertyFormatType = "observed",
        virtualPropertyIdentifiers: list[str] | None = None,
        aggregationMethod: (
            orchestrator.schema.virtual_property.PropertyAggregationMethodEnum | None
        ) = None,
    ) -> "DataFrame":
        """Returns a dataframe contain entities with at least one measured property"""

        from pandas import DataFrame

        references = self.measurementSpace.experimentReferences

        if property_type == "observed":
            return DataFrame(
                data=[
                    e.seriesRepresentation(
                        experimentReferences=references,
                        virtualTargetPropertyIdentifiers=virtualPropertyIdentifiers,
                        aggregationMethod=aggregationMethod,
                    )
                    for e in self.sampledEntities()
                    if len(e.observedPropertyValues) > 0
                ]
            )
        if property_type == "target":
            data = []
            for e in self.sampledEntities():
                if len(e.observedPropertyValues) > 0:
                    data.extend(
                        e.experimentSeries(
                            experimentReferences=references,
                            virtualTargetPropertyIdentifiers=virtualPropertyIdentifiers,
                            aggregationMethod=aggregationMethod,
                        )
                    )
            return DataFrame(data)
        raise ValueError(
            f"measuredEntitiesTable only supports the following "
            f"property_type: {DiscoverySpace.PropertyFormatType}"
        )

    def matchingEntitiesTable(
        self,
        property_type: PropertyFormatType = "observed",
        virtualPropertyIdentifiers: list[str] | None = None,
        aggregationMethod: (
            orchestrator.schema.virtual_property.PropertyAggregationMethodEnum | None
        ) = None,
    ) -> "DataFrame":
        """Returns a dataframe containing entities in the sample store that match the space definition.

        Notes:
        Only measurements that match the measurement space are output in the table

        The entities must have measurements from at least one of the experiments in the measurement space.
        This means that entities that match the entity-space but have no measurements from the measurement
        space are not output in the table

        Parameters:
            property_type: Controls if observed or target names are used to label properties
            virtualPropertyIdentifiers: An optional list of virtual property identifiers.
                These will replace the underlying property in the table
            aggregationMethod: Controls how to handle properties with multiple values (where
            no virtual property identifier is associated with them by previous parameter).
                By default, all values will be returned.
        """

        from pandas import DataFrame

        if property_type == "observed":
            return DataFrame(
                data=[
                    e.seriesRepresentation(
                        experimentReferences=self.measurementSpace.experimentReferences,
                        virtualTargetPropertyIdentifiers=virtualPropertyIdentifiers,
                        aggregationMethod=aggregationMethod,
                    )
                    for e in self.matchingEntities()
                    if len(e.observedPropertyValues) > 0
                    and not set(self.measurementSpace.experimentReferences).isdisjoint(
                        set(e.experimentReferences)
                    )
                ]
            )
        if property_type == "target":
            data = []
            for e in self.matchingEntities():
                if len(e.observedPropertyValues) > 0:
                    data.extend(
                        e.experimentSeries(
                            self.measurementSpace.experimentReferences,
                            virtualTargetPropertyIdentifiers=virtualPropertyIdentifiers,
                            aggregationMethod=aggregationMethod,
                        )
                    )
            return DataFrame(data)
        raise ValueError(
            f"matchingEntitiesTable only supports the following "
            f"property_type: {DiscoverySpace.PropertyFormatType}"
        )

    def storedEntitiesWithConstitutivePropertyValues(
        self,
        values: list[orchestrator.schema.property_value.PropertyValue],
        mode: typing.Literal["strict"] = "strict",
    ) -> list[None | orchestrator.schema.entity.Entity]:
        """Returns entities in the discoveryspace that have the given values for their constitutive properties and that are stored in the sample-store

        All entities returned will be strict members of this receivers entity space i.e. they will not have constitutive
        properties that are not in the discoveryspace's entity space.

        Raises:
            If there is a ExplicitEntitySpace then the following exceptions may be raised:

            ValueError: if any properties in constitutivePropertyValues are not part of the EntitySpace

            InconsistencyError: if values for all constitutive properties in the EntitySpace are given
            and more than one entity is found in the source that has exactly those constitutive properties.
            In an ExplicitSpace each point can only have one entity associated with it so there should be only
            one entity in the store that matches it
        """

        # If an entity-space is defined check that the request properties are actually in the space
        if self.entitySpace:
            requestedProperties = [v.property for v in values]
            try:
                definedProperties = [
                    c.descriptor() for c in self.entitySpace.constitutiveProperties
                ]
            except AttributeError:
                pass
            else:
                filtered = [
                    p for p in requestedProperties if p not in definedProperties
                ]

                if len(filtered) > 0:
                    raise ValueError(
                        f"Requested match against constitutive properties not in entity space definition: {filtered}"
                    )

        entities = self.sample_store.entitiesWithConstitutivePropertyValues(
            values=values
        )

        # The sample store returns any entity with the provided values.
        # now we have to filter for those in the space
        filteredEntities = [e for e in entities if self.entitySpace.isEntityInSpace(e)]

        # Check we don't have two entities with same id
        if len({e.identifier for e in filteredEntities}) != len(filteredEntities):
            raise SpaceInconsistencyError(
                f"Found more than one entity with same identifier in the sample store: {[e.identifier for e in filteredEntities]}."
            )

        return filteredEntities

    def entity_for_point(
        self,
        point: dict[str, tuple[Any]],
    ) -> Entity:
        """
        Returns an Entity instance for the given point.

        If this Entity exists in the DiscoverySpaces entity store that instance is returned.
        If not a new Entity instance is created. Note, this Entity instance is not added to the store.

        Parameters:
            point: A point in the discovery space as a dictionary of "constitutive property identifier":"value" pairs

        Exceptions:
            Raise ValueError if the point is not in the discovery space
        """

        property_identifiers = {
            cp.identifier for cp in self.entitySpace.constitutiveProperties
        }
        point_identifiers = set(point.keys())
        if diff := point_identifiers - property_identifiers:
            raise ValueError(
                f"Point {point} is not in space. It has values for additional properties, {diff}"
            )

        if diff := property_identifiers - point_identifiers:
            raise ValueError(
                f"Point {point} is not in space. It is missing values for properties, {diff}"
            )

        # Note if point contains additional properties this will just ignore them
        property_values = constitutive_property_values_from_point(
            point=point, properties=self.entitySpace.constitutiveProperties
        )

        try:
            entities = self.storedEntitiesWithConstitutivePropertyValues(
                values=property_values
            )
        except SpaceInconsistencyError:
            self.log.critical(
                "There are multiple entities with the same constitutive property value set"
            )
            raise
        else:
            entity = entities[0] if entities else None

        return entity or self.entitySpace.entity_for_point(point=point)

    #
    # Run/Operation Interface
    # Records runs on space
    #

    @property
    def metadataStore(self) -> "SQLStore":
        """Returns an interface to the metadata store used by the space"""

        return self._metadataStore

    @property
    def operations(self) -> "DataFrame":
        """Returns a table of all the operations executed on this space"""

        return self._metadataStore.getRelatedResourceIdentifiers(
            identifier=self.uri,
            kind=orchestrator.core.resources.CoreResourceKinds.OPERATION.value,
        )

    def addOperation(self, operation: OperationResource) -> None:
        """Add information on a new operation on the space

        Param:
            operation: The operation instance
        """

        self.log.debug(f"Adding run {operation}")

        self._metadataStore.addResourceWithRelationships(
            resource=operation, relatedIdentifiers=[self.uri]
        )

    def updateOperation(
        self,
        operationResource: OperationResource,
    ) -> None:
        """Update an operation resources metadata

        Params:
            operationResource: The operation resource to update.
        """

        self.log.info(f"Updating run {operationResource.identifier}")
        return self._metadataStore.updateResource(operationResource)

    @_perform_preflight_checks_for_sample_store_methods
    def complete_measurement_request_with_results_timeseries(
        self,
        operation_id: str,
        output_format: typing.Literal["target", "observed"],
        limit_to_properties: list[str] | None = None,
    ) -> "DataFrame":
        return self.sample_store.complete_measurement_request_with_results_timeseries(
            operation_id=operation_id,
            output_format=output_format,
            limit_to_properties=limit_to_properties,
        )

    @_perform_preflight_checks_for_sample_store_methods
    def entity_identifiers_in_operation(self, operation_id: str) -> set[str]:
        return self.sample_store.entity_identifiers_in_operation(
            operation_id=operation_id
        )

    @_perform_preflight_checks_for_sample_store_methods
    def experiments_in_operation(self, operation_id: str) -> list[Experiment]:
        return self.sample_store.experiments_in_operation(operation_id=operation_id)

    @_perform_preflight_checks_for_sample_store_methods
    def measurement_requests_for_operation(
        self, operation_id: str
    ) -> list[MeasurementRequest]:
        return self.sample_store.measurement_requests_for_operation(
            operation_id=operation_id
        )

    @_perform_preflight_checks_for_sample_store_methods
    def measurement_results_for_operation(
        self, operation_id: str
    ) -> list[MeasurementResult]:
        return self.sample_store.measurement_results_for_operation(
            operation_id=operation_id
        )
