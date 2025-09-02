# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import abc
import typing
from abc import ABC

import pydantic
from pydantic import ConfigDict

import orchestrator.core.samplestore.config
import orchestrator.utilities.location
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.property import ConstitutiveProperty
from orchestrator.schema.property_value import PropertyValue
from orchestrator.schema.request import MeasurementRequest


class SampleStore(abc.ABC):
    """Subclasses provide access to entities and may provide storage capability"""

    @classmethod
    @abc.abstractmethod
    def experimentCatalogFromReference(
        cls,
        reference: orchestrator.core.samplestore.config.SampleStoreReference | None,
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:  # pragma: nocover
        """ "
        Returns a catalog of the external experiments defined by a SampleStore
        Parameters:
            reference: An SampleStoreReference defining the SampleStore to access
            A reference is required as the class method may need identifier/parameters/storageLocation
            to access the information
        """

    @abc.abstractmethod
    def experimentCatalog(
        self,
    ) -> (
        orchestrator.modules.actuators.catalog.ExperimentCatalog | None
    ):  # pragma: nocover

        pass

    @property
    @abc.abstractmethod
    def entities(self) -> list[Entity]:  # pragma: nocover
        pass

    @property
    @abc.abstractmethod
    def numberOfEntities(self) -> int:  # pragma: nocover
        pass

    @abc.abstractmethod
    def containsEntityWithIdentifier(self, entity_id) -> bool:  # pragma: nocover
        pass

    def entitiesWithConstitutivePropertyValues(
        self, values: list[PropertyValue]
    ) -> list[Entity]:
        """Returns entities with which have the given constitutive property values

        Note: This is a non-optimized base method provided for convenience
        It will first get all entities then iterate over them.

        Params:
            values: A list of PropertyValue instances whose
            properties are Constitutive Properties

        Returns:
            A list of Entities in the receiver which have constitutivePropertyValues.
            If there are no matches the list will be empty.
        """

        def _same(entity, searchValues: list[PropertyValue]):
            # Does this entity have the same properties
            unmatchedProperties = [
                val
                for val in searchValues
                if entity.valueForProperty(val.property) is None
            ]
            if len(unmatchedProperties) == 0:
                unmatchedValues = [
                    val
                    for val in searchValues
                    if entity.valueForProperty(val.property).value != val.value
                ]

                return len(unmatchedValues) == 0
            return False

        all_entities = self.entities
        return [e for e in all_entities if _same(e, values)]

    @property
    @abc.abstractmethod
    def identifier(self) -> str:  # pragma: nocover
        """Return a unique identifier for this configuration of the sample store"""

    @property
    @abc.abstractmethod
    def config(self) -> typing.Any:  # pragma: nocover
        """Returns the parameter object used to initialise the receiver"""

    @property
    @abc.abstractmethod
    def location(
        self,
    ) -> orchestrator.utilities.location.ResourceLocation:  # pragma: nocover
        """Returns the location the sample store is stored in"""

    @staticmethod
    @abc.abstractmethod
    def validate_parameters(parameters=None) -> typing.Any:  # pragma: nocover
        """
        Validates the parameters to be passed to the class
        according to the concrete class's logic.

        The concrete class should return the parameters in the form their init should receive them
        """
        raise NotImplementedError(
            "Sample Stores must implement the validate_parameters method"
        )

    @staticmethod
    @abc.abstractmethod
    def storage_location_class() -> typing.Callable:  # pragma: nocover
        """
        Returns the ResourceLocation subclass to be used to instantiate the storageLocation parameter
        of the sample store's init method
        """
        raise NotImplementedError(
            "Sample Stores must implement the storage_location_class method"
        )


class PassiveSampleStore(SampleStore, ABC):
    """Subclasses provide access to entities but do not provide updates or store new entities"""

    @property
    def isPassive(self):
        return True


class ActiveSampleStore(SampleStore, ABC):
    """Subclasses provide access to entities but do not provide updates or store new entities"""

    @property
    def isPassive(self):
        return False

    @abc.abstractmethod
    def add_external_entities(self, entities: list[Entity]): ...  # pragma: nocover

    @abc.abstractmethod
    def addEntities(self, entities: list[Entity]):  # pragma: nocover
        """Add the entities to the sample store

        Check implementation for details on behaviour e.g. add v upsert.
        """

    @abc.abstractmethod
    def addMeasurement(self, measurementRequest: MeasurementRequest):  # pragma: nocover
        """Adds the results of a measurement to a set of entities

        Implementations of this method can require that the results have been already added to the
        Entities OR that measurementRequest.results is required instead.
        Check implementer for details.

        Parameters:
            measurementRequest: A MeasurementRequest instance

        """

    @abc.abstractmethod
    def entityWithIdentifier(
        self, entityIdentifier
    ) -> Entity | None:  # pragma: nocover
        # TODO: Probably this should also be supported by PassiveSampleStore
        pass

    @property
    @abc.abstractmethod
    def uri(self):  # pragma: nocover
        """Returns a URI for the Active Source"""

    @abc.abstractmethod
    def commit(self):  # pragma: nocover
        """Commits all the changes to the source and prevents any further changes"""


class MockParams(pydantic.BaseModel):

    numberOfEntities: int = pydantic.Field(default=100)
    model_config = ConfigDict(extra="forbid")


class ExperimentDescription(pydantic.BaseModel):
    experimentIdentifier: str = pydantic.Field(description="The name of the experiment")
    propertyMap: dict = pydantic.Field(
        description="A dictionary that maps the names of the properties exposed by the"
        " experiment to potential other names used for those properties by the sample store"
    )


class SampleStoreDescription(pydantic.BaseModel):
    experiments: list[ExperimentDescription] = pydantic.Field(
        default=[], description="A list describing the experiments in the source"
    )
    generatorIdentifier: str | None = pydantic.Field(
        default=None, description="The id of the entity generator"
    )

    @property
    def catalog(self):

        experiments = {}
        for desc in self.experiments:
            experiment = Experiment.experimentWithAbstractPropertyIdentifiers(
                identifier=desc.experimentIdentifier,
                actuatorIdentifier="replay",
                targetProperties=desc.propertyMap.keys(),
                requiredConstitutiveProperties=[
                    cp.identifier for cp in self.constitutiveProperties
                ],
            )
            experiments[experiment.identifier] = experiment

        return ExperimentCatalog(
            experiments=experiments, catalogIdentifier=self.generatorIdentifier
        )

    @property
    def experimentDescriptionMap(self):

        return {e.experimentIdentifier: e for e in self.experiments}

    @property
    def observedProperties(self):
        """Return all observed properties defined by the receiver"""

        observedProperties = []
        for e in self.catalog.experiments:
            observedProperties.extend(e.observedProperties)

        return observedProperties

    @property
    def constitutiveProperties(self) -> list[ConstitutiveProperty]:

        raise ValueError("Subclasses must implement this method")


class FailedToDecodeStoredEntityError(Exception):

    def __init__(
        self, entity_identifier: str, entity_representation: dict, cause: Exception
    ):
        self.entity_identifier = entity_identifier
        self.entity_representation = entity_representation
        self.cause = cause
        super().__init__(
            f"Unable to decode representation for entity {entity_identifier}.\n\n"
            f"Representation was: {entity_representation}.\n\n"
            f"Error was: {cause}"
        )


class FailedToDecodeStoredMeasurementResultForEntityError(Exception):

    def __init__(
        self, entity_identifier: str, result_representation: dict, cause: Exception
    ):
        self.entity_identifier = entity_identifier
        self.result_representation = result_representation
        self.cause = cause
        super().__init__(
            f"Unable to decode a measurement result for entity {entity_identifier}.\n\n"
            f"Result representation was: {result_representation}.\n\n"
            f"Error was: {cause}"
        )
