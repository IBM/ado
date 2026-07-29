# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import contextlib
import logging
import os
import typing
import warnings
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any

import ado.core.discoveryspace.resource
import ado.core.metadata
import ado.core.resources
import ado.core.samplestore.base
import ado.schema.entity
import ado.schema.measurementspace
import ado.schema.property_value
import ado.schema.virtual_property
import ado.utilities.logging
from ado.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
    DiscoverySpaceProperties,
)
from ado.core.operation.config import DiscoveryOperationEnum
from ado.core.operation.resource import OperationResource
from ado.core.resources import ADOResourceReference, CoreResourceKinds
from ado.metastore.project import ProjectContext
from ado.modules.actuators.catalog import ActuatorCatalogExtension
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.entity import Entity
from ado.schema.entityspace import (
    EntitySpaceRepresentation,
)
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace
from ado.schema.property_value import constitutive_property_values_from_point
from ado.schema.request import MeasurementRequest
from ado.schema.result import MeasurementResult

if typing.TYPE_CHECKING:
    from pandas import DataFrame
    from rich.console import RenderableType

    from ado.metastore.sqlstore import SQLResourceStore, SQLStore

FORMAT = ado.utilities.logging.FORMAT
LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, format=FORMAT)

moduleLogger = logging.getLogger("discoveryspace")

SCRIPT_OPERATION_EXECUTION_LABEL = "script"
SCRIPT_OPERATION_LABEL_KEY = "execution"


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
        self: "DiscoverySpace",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401

        import ado.core.samplestore.sql

        if not isinstance(self.sample_store, ado.core.samplestore.sql.SQLSampleStore):
            raise ValueError(
                "The complete_measurement_request_with_results_timeseries method "
                "requires the use of an SQLSampleStore"
            )

        operation_id = kwargs.get("operation_id") or args[0]

        # Skip the DB round-trip when we've already verified this operation
        # belongs to this space (e.g. the space was built via from_operation_id).
        if operation_id not in self._verified_operation_ids:
            operation_spaces = self._metadataStore.getResource(
                identifier=operation_id,
                kind=CoreResourceKinds.OPERATION,
                raise_error_if_no_resource=True,
            ).config.spaces

            if self.uri not in operation_spaces:
                raise ValueError(
                    f"Operation {operation_id} does not belong to space {self.uri}; "
                    f"its spaces are: {operation_spaces}"
                )

            self._verified_operation_ids.add(operation_id)

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
        metadata_store: "SQLResourceStore | None" = None,
        samplestore_resource: "ado.core.SampleStoreResource | None" = None,
        sample_store: "ado.core.samplestore.base.SampleStore | None" = None,
        load_experiment_catalog: bool = True,
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
            metadata_store: Optional SQLResourceStore instance to reuse. If None, a new instance will be created.
            samplestore_resource: Optional pre-fetched SampleStoreResource. When provided the metastore
                round-trip to fetch the samplestore is skipped. Ignored when *sample_store* is provided.
            sample_store: An already-instantiated SampleStore to use directly. When provided, both
                the metastore round-trip and the ``SampleStore.from_resource`` call are skipped,
                and the same object is reused — preserving any in-memory entity cache.
                Takes precedence over *samplestore_resource*.
            load_experiment_catalog: When ``True`` (default) experiments from the samplestore
                catalog are also registered with the global actuator registry. When ``False``,
                the catalog is still loaded locally for measurement-space resolution but not
                registered globally. Use ``False`` for read-only paths (e.g. CLI show commands)
                to avoid conflicting with experiments already registered in the same process.

        """

        from ado.core.samplestore.base import SampleStore

        if metadata_store is None:
            metadata_store = ado.metastore.sqlstore.SQLResourceStore(
                project_context=project_context
            )

        entitySpace = None

        if sample_store is None:
            sample_store = (
                SampleStore.from_resource(samplestore_resource)
                if samplestore_resource is not None
                else SampleStore.from_identifier(
                    identifier=conf.sampleStoreIdentifier, metastore=metadata_store
                )
            )

        if conf.entitySpace is not None:
            entitySpace = EntitySpaceRepresentation.representationFromConfiguration(
                conf.entitySpace
            )

        ## Load external experiments from the sample store for measurement-space
        ## resolution. Only register them globally when requested.
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
                if load_experiment_catalog:
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
            ado.schema.measurementspace.MeasurementSpaceConfiguration,
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
            metadata_store=metadata_store,
        )

    @classmethod
    def from_stored_configuration(
        cls,
        project_context: ProjectContext,
        space_identifier: str,
        metadata_store: "SQLResourceStore | None" = None,
        space_resource: "ado.core.DiscoverySpaceResource | None" = None,
        samplestore_resource: "ado.core.SampleStoreResource | None" = None,
        load_experiment_catalog: bool = True,
    ) -> "DiscoverySpace":
        """Creates a DiscoverySpace from a stored space identifier.

        Args:
            project_context: Project context used to connect to the metadata store.
            space_identifier: Identifier of the stored DiscoverySpace resource.
            metadata_store: Optional SQLResourceStore instance to reuse.
            space_resource: Optional pre-fetched DiscoverySpaceResource. When provided
                the metastore round-trip to fetch the space is skipped.
            samplestore_resource: Optional pre-fetched SampleStoreResource. Forwarded
                to :meth:`from_configuration` to skip the samplestore round-trip.
            load_experiment_catalog: Forwarded to :meth:`from_configuration`.
        """
        from ado.metastore.sqlstore import SQLStore

        moduleLogger.debug("Accessing discovery space metadata store")
        if metadata_store is None:
            metadata_store = SQLStore(project_context=project_context)

        if space_resource is None:
            moduleLogger.debug(
                f"Retrieving configuration for discovery space {space_identifier}"
            )
            space_resource = metadata_store.getResource(
                identifier=space_identifier,
                kind=CoreResourceKinds.DISCOVERYSPACE,
                raise_error_if_no_resource=True,
            )
        conf = space_resource.config

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
            metadata_store=metadata_store,
            samplestore_resource=samplestore_resource,
            load_experiment_catalog=load_experiment_catalog,
        )

    @classmethod
    def from_operation_id(
        cls,
        operation_id: str,
        project_context: ProjectContext,
        metadata_store: "SQLResourceStore | None" = None,
    ) -> "DiscoverySpace":
        """
        Creates a DiscoverySpace instance of the class from the given operation id and project context.

        Args:
            operation_id (str): The operation id to be used for finding the space identifier.
            project_context (ProjectContext): The project context to be used for creating the discovery space.
            metadata_store: Optional SQLResourceStore instance to reuse. If None, a new instance will be created.

        Returns:
            DiscoverySpace: The newly created discovery space instance.

        Raises:
            ResourceDoesNotExistError: If the specified operation or related space do not exist.
            NoRelatedResourcesError: If no sample store is associated with the specified operation or related space.
        """
        from ado.metastore.sqlstore import SQLStore

        if metadata_store is None:
            metadata_store = SQLStore(project_context=project_context)

        # Fetch the operation, its space, and the space's samplestore in a
        # single SQL JOIN rather than three sequential round-trips.
        # FIXME AP 12/06/2025:
        # We are using the first space - which may become a problem in the future
        _, space_resource, samplestore_resource = (
            metadata_store.get_resource_and_producers(
                identifier=operation_id,
                kind=CoreResourceKinds.OPERATION,
                chain=[
                    ("$.config.spaces[0]", CoreResourceKinds.DISCOVERYSPACE),
                    ("$.config.sampleStoreIdentifier", CoreResourceKinds.SAMPLESTORE),
                ],
                raise_error_if_no_resource=True,
            )
        )

        space = cls.from_stored_configuration(
            project_context=project_context,
            space_identifier=space_resource.identifier,
            metadata_store=metadata_store,
            space_resource=space_resource,
            samplestore_resource=samplestore_resource,
            load_experiment_catalog=False,
        )
        space._verified_operation_ids.add(operation_id)
        return space

    def __init__(
        self,
        project_context: ProjectContext,
        identifier: str | None = None,
        sample_store: ado.core.samplestore.base.ActiveSampleStore | None = None,
        entitySpace: EntitySpaceRepresentation | None = None,
        measurementSpace: MeasurementSpace | None = None,
        properties: (
            ado.core.discoveryspace.config.DiscoverySpaceProperties | None
        ) = None,
        metadata: ado.core.metadata.ConfigurationMetadata | None = None,
        metadata_store: "ado.metastore.sqlstore.SQLStore | None" = None,
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

        # Access metadata store - reuse provided instance if available
        if metadata_store is None:
            from ado.metastore.sqlstore import SQLStore

            self._metadataStore = SQLStore(project_context=project_context)
        else:
            self._metadataStore = metadata_store

        self._identifier = (
            identifier
            if identifier is not None
            else f"space-{str(uuid.uuid4())[:6]}-{self._sample_store.identifier}"
        )

        # Operation IDs that have already passed the preflight ownership check.
        # Pre-populated by from_operation_id to avoid a redundant DB round-trip.
        self._verified_operation_ids: set[str] = set()

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
    def reference(self) -> ADOResourceReference:
        """Return a metastore reference for this discovery space."""

        return ADOResourceReference(
            identifier=self.uri,
            kind=CoreResourceKinds.DISCOVERYSPACE,
        )

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
    ) -> ado.core.samplestore.base.ActiveSampleStore:
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
        # (a) an explicit import of ado.actuators.base (b) information on all dynamic actuator modules used.

        metadata = (
            self._metadata
            if self._metadata is not None
            else ado.core.metadata.ConfigurationMetadata()
        )

        return DiscoverySpaceConfiguration(
            sampleStoreIdentifier=self.sample_store.identifier,
            entitySpace=entitySpaceConf,
            experiments=self.measurementSpace.selfContainedConfig,
            metadata=metadata,
        )

    def _build_provenance(
        self,
    ) -> "ado.core.discoveryspace.resource.DiscoverySpaceProvenanceInfo":
        """Resolve package provenance for all actuators and custom experiments.

        Returns:
            DiscoverySpaceProvenanceInfo mapping actuators and custom experiments
            to the distributions that provided them at space creation time.
        """
        from ado.core.discoveryspace.resource import (
            DiscoverySpaceProvenanceInfo,
        )
        from ado.core.metadata import PackageProvenance
        from ado.modules.actuators.registry import ActuatorRegistry

        registry = ActuatorRegistry.globalRegistry()
        actuators: dict[str, PackageProvenance] = {}
        custom_experiments: dict[str, PackageProvenance] = {}

        for experiment in self.measurementSpace.experiments:
            actuator_id = experiment.actuatorIdentifier

            # Per-actuator provenance (deduplicated)
            if actuator_id not in actuators:
                provenance = registry.provenance_for_actuator(actuator_id)
                if provenance is not None:
                    actuators[actuator_id] = provenance

            # Per-custom-experiment provenance
            if actuator_id == "custom_experiments":
                module_conf = experiment.metadata.get("module")
                if module_conf is not None:
                    provenance = PackageProvenance.from_module_conf(module_conf)
                    if provenance is not None:
                        custom_experiments[experiment.identifier] = provenance

        return DiscoverySpaceProvenanceInfo(
            actuators=actuators,
            customExperiments=custom_experiments,
        )

    @property
    def resource(
        self,
    ) -> ado.core.discoveryspace.resource.DiscoverySpaceResource:

        return ado.core.discoveryspace.resource.DiscoverySpaceResource(
            identifier=self._identifier,
            config=self.config,
            provenance=self._build_provenance(),
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

    def sampledEntities(self, *, require_measurements: bool = True) -> list[Entity]:
        """Returns the entities sampled so far in the space.

        Args:
            require_measurements: When ``True`` (default), measurement results
                are guaranteed to be attached to each entity.  Pass ``False`` to
                load only constitutive properties (measurement results may still
                be present if already loaded).
        """

        operation_ids = self.operations

        if not operation_ids:
            return []

        entity_ids = set(
            self.sample_store.entity_identifiers_in_operations(operation_ids)
        )
        sampled_entities = self.sample_store.get_entities(
            identifiers=entity_ids, require_measurements=require_measurements
        )

        # TODO: Consider removing isEntitySpace check
        # The additional check of isEntityInSpace should not be required if things are working correctly
        # However if an entity was incorrectly sampled during an operation, due to a bug say, this will correct for it
        return [e for e in sampled_entities if self.entitySpace.isEntityInSpace(e)]

    def matchingEntities(self, *, require_measurements: bool = True) -> list[Entity]:
        """Returns all entities in the sample store that match the space.

        Args:
            require_measurements: When ``True`` (default), measurement results
                are guaranteed to be attached to each entity.  Pass ``False`` to
                load only constitutive properties (measurement results may still
                be present if already loaded).

        If
        - ExplicitEntitySpace defined -> filter on the space
        - No space defined -> implies the space entities == all entities in the source.
        """

        # Get all entities in the store
        all_entities = self.sample_store.get_entities(
            require_measurements=require_measurements
        )
        if self.entitySpace is None:
            return all_entities

        if not self.entitySpace.isDiscreteSpace:
            return [e for e in all_entities if self.entitySpace.isEntityInSpace(e)]

        entities = []
        for entity in all_entities:
            if self.entitySpace.isEntityInSpace(entity):
                entities.append(entity)
                if len(entities) == self.entitySpace.size:
                    break

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
            ado.schema.virtual_property.PropertyAggregationMethodEnum | None
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
            ado.schema.virtual_property.PropertyAggregationMethodEnum | None
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
        values: list[ado.schema.property_value.PropertyValue],
        mode: typing.Literal["strict"] = "strict",
    ) -> list[ado.schema.entity.Entity | None]:
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
    def operations(self) -> set[str]:
        """Returns the identifiers of all operations executed on this space"""

        return self._metadataStore.get_resources_by_relationship(
            kind=ado.core.resources.CoreResourceKinds.DISCOVERYSPACE,
            identifier=self.uri,
            hierarchy_direction="down",
            max_hops=1,
            identifiers_only=True,
        ).get(ado.core.resources.CoreResourceKinds.OPERATION, set())

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

    @contextlib.contextmanager
    def operation_context(
        self,
        name: str,
        description: str | None = None,
        metadata: dict | None = None,
        operation_type: DiscoveryOperationEnum = DiscoveryOperationEnum.EXPLORE,
        provenance: "ado.core.metadata.PackageProvenance | None" = None,
    ) -> Iterator[str]:
        """Context manager that registers a script operation and manages its lifecycle.

        Creates an OperationResource linked to this space, appends STARTED before
        yielding the operation_id, and writes FINISHED/SUCCESS or FINISHED/FAIL on exit.
        The operation_id should be passed as ``requesterid`` to Actuators execute
        or submit methods

        Args:
            name: Human-readable script name stored in the operation configuration.
            description: Optional description for the operation metadata.
            metadata: Optional extra metadata fields merged into ConfigurationMetadata.
            operation_type: Semantic type for the operation (e.g. EXPLORE for explore scripts).
                Script provenance is always recorded on metadata labels under
                ``execution: script``.
            provenance: Optional Python distribution provenance for the script module.
                When provided, stored under ``provenance.operators`` keyed by the
                script operator identifier.

        Yields:
            The operation resource identifier.

        Raises:
            RuntimeError: If the discovery space has no metadata store.
        """
        if self._metadataStore is None:
            raise RuntimeError(
                "DiscoverySpace.operation_context requires a metadata store; "
                "load the space from stored configuration first."
            )

        from ado.core.metadata import ConfigurationMetadata
        from ado.core.operation.config import (
            DiscoveryOperationConfiguration,
            DiscoveryOperationResourceConfiguration,
            ScriptOperatorConf,
        )
        from ado.core.operation.resource import (
            OperationExitStateEnum,
            OperationProvenanceInfo,
            OperationResource,
            OperationResourceEventEnum,
            OperationResourceStatus,
        )

        script_module = ScriptOperatorConf(name=name, operationType=operation_type)
        extra_metadata = dict(metadata or {})
        user_labels = extra_metadata.pop("labels", None) or {}
        config_metadata = ConfigurationMetadata(
            name=name,
            description=description,
            labels={
                SCRIPT_OPERATION_LABEL_KEY: SCRIPT_OPERATION_EXECUTION_LABEL,
                **user_labels,
            },
        )
        for key, value in extra_metadata.items():
            setattr(config_metadata, key, value)

        operation_payload = DiscoveryOperationResourceConfiguration(
            operation=DiscoveryOperationConfiguration(
                module=script_module,
                parameters={},
            ),
            metadata=config_metadata,
            inputs={"discoverySpace": self.reference},
        )

        if provenance is None:
            final_provenance = OperationProvenanceInfo(operators={})
        else:
            final_provenance = OperationProvenanceInfo(
                operators={script_module.operatorIdentifier: provenance},
            )

        operation = OperationResource(
            operationType=script_module.operationType,
            operatorIdentifier=script_module.operatorIdentifier,
            config=operation_payload,
            provenance=final_provenance,
        )

        self.addOperation(operation)
        self._verified_operation_ids.add(operation.identifier)

        try:
            operation.status.append(
                OperationResourceStatus(event=OperationResourceEventEnum.STARTED)
            )
            self.updateOperation(operation)
            yield operation.identifier
            operation.status.append(
                OperationResourceStatus(
                    event=OperationResourceEventEnum.FINISHED,
                    exit_state=OperationExitStateEnum.SUCCESS,
                )
            )
        except Exception:
            operation.status.append(
                OperationResourceStatus(
                    event=OperationResourceEventEnum.FINISHED,
                    exit_state=OperationExitStateEnum.FAIL,
                )
            )
            raise
        finally:
            self.updateOperation(operation)

    @_perform_preflight_checks_for_sample_store_methods
    def complete_measurement_request_with_results_timeseries(
        self,
        operation_id: str,
        output_format: typing.Literal["target", "observed"],
        limit_to_properties: list[str] | None = None,
        aggregation_method: (
            ado.schema.virtual_property.PropertyAggregationMethodEnum | None
        ) = None,
    ) -> "DataFrame":
        return self.sample_store.complete_measurement_request_with_results_timeseries(
            operation_id=operation_id,
            output_format=output_format,
            limit_to_properties=limit_to_properties,
            aggregation_method=aggregation_method,
        )

    @_perform_preflight_checks_for_sample_store_methods
    def entity_identifiers_in_operations(
        self,
        operation_ids: str | set[str],
        group_by_operation: bool = False,
    ) -> set[str] | dict[str, set[str]]:
        """Return entity identifiers sampled in the given operation(s).

        Args:
            operation_ids: A single operation identifier or a set of operation
                identifiers to look up entity identifiers for.
            group_by_operation: When True, return a dict mapping each operation
                ID to its set of entity identifiers. When False (default),
                return a flat set of entity identifiers across all operations.

        Returns:
            A flat set of entity identifier strings when group_by_operation is
            False, or a dict mapping operation ID to set of entity identifiers
            when group_by_operation is True.
        """
        return self.sample_store.entity_identifiers_in_operations(
            operation_ids=operation_ids,
            group_by_operation=group_by_operation,
        )

    @_perform_preflight_checks_for_sample_store_methods
    def entity_identifiers_in_operation(
        self, operation_ids: str | set[str]
    ) -> set[str]:
        """Deprecated: use entity_identifiers_in_operations instead."""
        warnings.warn(
            "entity_identifiers_in_operation is deprecated, use entity_identifiers_in_operations instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.entity_identifiers_in_operations(operation_ids)

    @_perform_preflight_checks_for_sample_store_methods
    def experiments_in_operation(self, operation_id: str) -> list[Experiment]:
        return self.sample_store.experiments_in_operation(operation_id=operation_id)

    @_perform_preflight_checks_for_sample_store_methods
    def measurement_requests_for_operation(
        self, operation_id: str | set[str]
    ) -> list[MeasurementRequest] | dict[str, list[MeasurementRequest]]:
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

    @_perform_preflight_checks_for_sample_store_methods
    def operation_entity_statistics(self, operation_id: str) -> dict[str, int]:
        """
        Compute entity-level statistics for an operation using SQL aggregation.

        Returns a dictionary with entity counts for the operation.
        """
        import ado.core.samplestore.sql

        if isinstance(self.sample_store, ado.core.samplestore.sql.SQLSampleStore):
            return self.sample_store.operation_entity_statistics(
                operation_id=operation_id
            )
        # Fallback for non-SQL sample stores: fetch all results and count in Python
        measurement_results = self.measurement_results_for_operation(
            operation_id=operation_id
        )
        from ado.schema.result import ValidMeasurementResult

        entities_with_all_successful_measurements = {
            result.entityIdentifier for result in measurement_results
        }
        entities_with_at_least_one_successful_measurement = set()
        for measurement_result in measurement_results:
            if isinstance(measurement_result, ValidMeasurementResult):
                entities_with_at_least_one_successful_measurement.add(
                    measurement_result.entityIdentifier
                )
                continue
            entities_with_all_successful_measurements.discard(
                measurement_result.entityIdentifier
            )

        return {
            "entities_with_all_successful_measurements": len(
                entities_with_all_successful_measurements
            ),
            "entities_with_at_least_one_successful_measurement": len(
                entities_with_at_least_one_successful_measurement
            ),
            "total_entities": len(
                {result.entityIdentifier for result in measurement_results}
            ),
        }

    @_perform_preflight_checks_for_sample_store_methods
    def operation_measurement_statistics(
        self, operation_ids: set[str] | None = None
    ) -> "list[ado.core.operation.stats.OperationMeasurementStatistics]":
        """Compute aggregated measurement statistics for one or more operations.

        Delegates to the SQL implementation for SQL-backed stores. For all
        other stores, falls back to a Python implementation that iterates the
        measurement requests per operation.

        Args:
            operation_ids: Set of operation identifiers to aggregate. Pass
                ``None`` to aggregate across all operations in the store.
                Passing an empty set raises ``ValueError``.

        Returns:
            A list of OperationMeasurementStatistics instances, one per
            operation found in the store.

        Raises:
            ValueError: If ``operation_ids`` is an empty set.
        """
        if operation_ids is not None and len(operation_ids) == 0:
            raise ValueError("operation_ids must be a non-empty set or None")

        import ado.core.samplestore.sql
        from ado.core.operation.stats import OperationMeasurementStatistics

        if isinstance(self.sample_store, ado.core.samplestore.sql.SQLSampleStore):
            return self.sample_store.operation_measurement_statistics(
                operation_ids=operation_ids
            )

        # Python fallback for non-SQL stores
        from ado.schema.request import MeasurementRequestStateEnum
        from ado.schema.result import ValidMeasurementResult

        # Determine which operation IDs to iterate
        ids_to_process: set[str] = (
            self.operations if operation_ids is None else operation_ids
        )

        result_list: list[OperationMeasurementStatistics] = []
        for op_id in ids_to_process:
            requests = self.measurement_requests_for_operation(operation_id=op_id)

            total_requests = len(requests)
            failed_requests = sum(
                1 for r in requests if r.status == MeasurementRequestStateEnum.FAILED
            )
            successful_requests = sum(
                1 for r in requests if r.status == MeasurementRequestStateEnum.SUCCESS
            )

            total_results = 0
            successful_results = 0
            failed_results = 0
            measured_entity_ids: set[str] = set()

            for request in requests:
                for result in request.measurements:
                    total_results += 1
                    if isinstance(result, ValidMeasurementResult):
                        successful_results += 1
                    else:
                        failed_results += 1
                    measured_entity_ids.add(result.entityIdentifier)

            result_list.append(
                OperationMeasurementStatistics(
                    operation_id=op_id,
                    total_requests=total_requests,
                    failed_requests=failed_requests,
                    successful_requests=successful_requests,
                    total_results=total_results,
                    successful_results=successful_results,
                    failed_results=failed_results,
                    measured_entities=len(measured_entity_ids),
                )
            )

        return result_list

    def space_statistics(
        self, lightweight_only: bool = False
    ) -> "ado.core.discoveryspace.stats.DiscoverySpaceStatistics":
        """Compute statistics for this discovery space.

        Delegates to
        :func:`~ado.core.discoveryspace.stats.space_statistics_for_spaces`
        for a single space.

        Args:
            lightweight_only: When ``True`` skip all Python-side computation
                and return ``None`` for the heavy fields.

        Returns:
            :class:`~ado.core.discoveryspace.stats.DiscoverySpaceStatistics`
        """
        from ado.core.discoveryspace.stats import space_statistics_for_spaces

        return space_statistics_for_spaces([self], lightweight_only=lightweight_only)[
            self.uri
        ]
