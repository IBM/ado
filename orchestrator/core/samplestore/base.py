# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import abc
import typing
from abc import ABC
from typing import Annotated, Literal

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

if typing.TYPE_CHECKING:
    from orchestrator.schema.observed_property import ObservedProperty


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
    def containsEntityWithIdentifier(self, entity_id: str) -> bool:  # pragma: nocover
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

        def _same(entity: Entity, searchValues: list[PropertyValue]) -> bool:
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
    def config(self) -> typing.Any:  # noqa: ANN401 # pragma: nocover
        """Returns the parameter object used to initialise the receiver"""

    @property
    @abc.abstractmethod
    def location(
        self,
    ) -> orchestrator.utilities.location.ResourceLocation:  # pragma: nocover
        """Returns the location the sample store is stored in"""

    @staticmethod
    @abc.abstractmethod
    def validate_parameters(
        parameters: dict,
    ) -> typing.Any:  # noqa: ANN401 # pragma: nocover
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
    def isPassive(self) -> bool:
        return True


class ActiveSampleStore(SampleStore, ABC):
    """Subclasses provide access to entities but do not provide updates or store new entities"""

    @property
    def isPassive(self) -> bool:
        return False

    @abc.abstractmethod
    def add_external_entities(
        self, entities: list[Entity]
    ) -> None: ...  # pragma: nocover

    @abc.abstractmethod
    def addEntities(self, entities: list[Entity]) -> None:  # pragma: nocover
        """Add the entities to the sample store

        Check implementation for details on behaviour e.g. add v upsert.
        """

    @abc.abstractmethod
    def addMeasurement(
        self, measurementRequest: MeasurementRequest
    ) -> None:  # pragma: nocover
        """Adds the results of a measurement to a set of entities

        Implementations of this method can require that the results have been already added to the
        Entities OR that measurementRequest.results is required instead.
        Check implementer for details.

        Parameters:
            measurementRequest: A MeasurementRequest instance

        """

    @abc.abstractmethod
    def entityWithIdentifier(
        self, entityIdentifier: str
    ) -> Entity | None:  # pragma: nocover
        # TODO: Probably this should also be supported by PassiveSampleStore
        pass

    @property
    @abc.abstractmethod
    def uri(self) -> str:  # pragma: nocover
        """Returns a URI for the Active Source"""

    @abc.abstractmethod
    def commit(self) -> None:  # pragma: nocover
        """Commits all the changes to the source and prevents any further changes"""


class MockParams(pydantic.BaseModel):
    numberOfEntities: Annotated[int, pydantic.Field()] = 100
    model_config = ConfigDict(extra="forbid")


class ExperimentDescription(pydantic.BaseModel):
    """Base class for experiment descriptions in sample stores"""

    experimentIdentifier: Annotated[
        str, pydantic.Field(description="The name of the experiment")
    ]
    actuatorIdentifier: Annotated[
        str,
        pydantic.Field(description="The actuator that provides this experiment"),
    ]
    observedPropertyMap: Annotated[
        dict[str, str] | list[str],
        pydantic.Field(
            description="Mapping of property names from the experiment to column names in the sample store. "
            "Use a dictionary (e.g., {'experiment_prop': 'store_column'}) when names differ, "
            "or a simple list (e.g., ['prop1', 'prop2']) when names are identical in both places."
        ),
    ]
    constitutivePropertyMap: Annotated[
        dict[str, str] | list[str],
        pydantic.Field(
            description="Mapping of property names from the experiment to column names in the sample store. "
            "Use a dictionary (e.g., {'experiment_prop': 'store_column'}) when names differ, "
            "or a simple list (e.g., ['prop1', 'prop2']) when names are identical in both places."
        ),
    ]

    @pydantic.field_validator(
        "observedPropertyMap", "constitutivePropertyMap", mode="before"
    )
    @classmethod
    def convert_list_to_dict(cls, value: dict[str, str] | list[str]) -> dict[str, str]:
        """Convert list format to dict format where keys equal values"""
        if isinstance(value, list):
            return {item: item for item in value}
        return value

    @pydantic.field_serializer("observedPropertyMap", "constitutivePropertyMap")
    def serialize_property_map_as_list_if_identical(
        self, value: dict[str, str]
    ) -> dict[str, str] | list[str]:
        """Serialize property map as list if keys and values are identical"""
        if isinstance(value, dict) and all(key == val for key, val in value.items()):
            # Return as list preserving insertion order
            return list(value.keys())
        return value

    @property
    def source_observed_property_identifiers(self) -> list[str]:
        """Returns the sample store column names for the observed properties."""
        return list(self.observedPropertyMap.values())

    @property
    def source_constitutive_property_identifiers(self) -> list[str]:
        """Returns the sample store column names for the constitutive properties."""
        return list(self.constitutivePropertyMap.values())

    @property
    def experiment(self) -> Experiment:
        """Returns an Experiment object configured with this description's properties.

        The experiment uses the property names (keys from the property maps) as its
        target observed properties and required constitutive properties.
        """
        return Experiment.experimentWithAbstractPropertyIdentifiers(
            identifier=self.experimentIdentifier,
            actuatorIdentifier=self.actuatorIdentifier,
            targetProperties=self.observedPropertyMap.keys(),
            requiredConstitutiveProperties=self.constitutivePropertyMap.keys(),
        )


class ExternalExperimentDescription(ExperimentDescription):
    """Experiment description for external (replay) experiments"""

    model_config = pydantic.ConfigDict(extra="forbid")

    actuatorIdentifier: Annotated[
        Literal["replay"],
        pydantic.Field(
            description="External experiments always use the 'replay' actuator",
        ),
    ] = "replay"


class InternalExperimentDescription(ExperimentDescription):
    """Experiment description for internal experiments with auto-inference"""

    model_config = pydantic.ConfigDict(extra="forbid")

    observedPropertyMap: Annotated[
        dict[str, str] | list[str] | None,
        pydantic.Field(
            description="Mapping of property names from the experiment to column names in the sample store. "
            "Use a dictionary (e.g., {'experiment_prop': 'store_column'}) when names differ, "
            "or a simple list (e.g., ['prop1', 'prop2']) when names are identical in both places.",
        ),
    ] = None
    constitutivePropertyMap: Annotated[
        dict[str, str] | list[str] | None,
        pydantic.Field(
            description="Mapping of property names from the experiment to column names in the sample store. "
            "Use a dictionary (e.g., {'experiment_prop': 'store_column'}) when names differ, "
            "or a simple list (e.g., ['prop1', 'prop2']) when names are identical in both places.",
        ),
    ] = None
    propertyFormat: Annotated[
        Literal["target", "observed"],
        pydantic.Field(
            description="Whether source names for observed properties map to experiment target property (default) or observed property identifiers.",
        ),
    ] = "target"

    @pydantic.model_validator(mode="after")
    def infer_and_validate_property_maps(self) -> "InternalExperimentDescription":
        """Infer property maps from experiment definition and validate"""
        from orchestrator.modules.actuators.registry import ActuatorRegistry
        from orchestrator.schema.reference import ExperimentReference

        registry = ActuatorRegistry.globalRegistry()

        # Get the experiment definition (will raise exception if not found)
        exp_ref = ExperimentReference(
            experimentIdentifier=self.experimentIdentifier,
            actuatorIdentifier=self.actuatorIdentifier,
        )
        experiment = registry.experimentForReference(exp_ref)

        # Infer observedPropertyMap if not provided
        # Keys are always target property identifiers
        # Values depend on propertyFormat
        if not self.observedPropertyMap:
            if self.propertyFormat == "target":
                # Map target property identifiers to themselves (column names match target identifiers)
                self.observedPropertyMap = {
                    prop.identifier: prop.identifier
                    for prop in experiment.targetProperties
                }
            else:  # "observed"
                # Map target property identifiers to observed property identifiers (column names match observed identifiers)
                self.observedPropertyMap = {
                    obs_prop.targetProperty.identifier: obs_prop.identifier
                    for obs_prop in experiment.observedProperties
                }

        # Validate observedPropertyMap keys (always target property identifiers)
        valid_target_identifiers = {
            prop.identifier for prop in experiment.targetProperties
        }
        provided_keys = set(self.observedPropertyMap.keys())

        # Check for invalid keys (we don't require all target properties to be present)
        invalid_keys = provided_keys - valid_target_identifiers
        if invalid_keys:
            raise ValueError(
                f"observedPropertyMap contains invalid target property identifiers: "
                f"{sorted(invalid_keys)}. Valid identifiers: {sorted(valid_target_identifiers)}"
            )

        # Infer constitutivePropertyMap if not provided
        if not self.constitutivePropertyMap:
            # Map constitutive property identifiers to themselves
            self.constitutivePropertyMap = {
                prop.identifier: prop.identifier
                for prop in experiment.requiredConstitutiveProperties
            }

        # Validate constitutivePropertyMap keys
        valid_constitutive_identifiers = {
            prop.identifier for prop in experiment.requiredConstitutiveProperties
        }
        provided_const_keys = set(self.constitutivePropertyMap.keys())

        # Check for invalid keys
        invalid_const_keys = provided_const_keys - valid_constitutive_identifiers
        if invalid_const_keys:
            raise ValueError(
                f"constitutivePropertyMap contains invalid constitutive property identifiers: "
                f"{sorted(invalid_const_keys)}. Valid identifiers: {sorted(valid_constitutive_identifiers)}"
            )

        # Check that all required keys are present
        missing_const_keys = valid_constitutive_identifiers - provided_const_keys
        if missing_const_keys:
            raise ValueError(
                f"constitutivePropertyMap is missing required constitutive property identifiers: "
                f"{sorted(missing_const_keys)}"
            )

        return self

    @property
    def experiment(self) -> Experiment:
        """Returns the experiment from the actuator registry"""
        from orchestrator.modules.actuators.registry import ActuatorRegistry
        from orchestrator.schema.reference import ExperimentReference

        registry = ActuatorRegistry.globalRegistry()
        exp_ref = ExperimentReference(
            experimentIdentifier=self.experimentIdentifier,
            actuatorIdentifier=self.actuatorIdentifier,
        )
        return registry.experimentForReference(exp_ref)


def source_experiment_description_discriminator(
    desc: dict | ExternalExperimentDescription | InternalExperimentDescription,
) -> str:
    """Discriminator function for SourceExperimentDescription union"""
    if isinstance(desc, ExternalExperimentDescription):
        return "External"
    if isinstance(desc, InternalExperimentDescription):
        return "Internal"
    if isinstance(desc, dict):
        actuator_id = desc.get("actuatorIdentifier", "replay")
        return "External" if actuator_id == "replay" else "Internal"

    raise ValueError(
        f"Unable to determine source experiment description type for desc: {desc}"
    )


SourceExperimentDescription = Annotated[
    Annotated[ExternalExperimentDescription, pydantic.Tag("External")]
    | Annotated[InternalExperimentDescription, pydantic.Tag("Internal")],
    pydantic.Discriminator(source_experiment_description_discriminator),
]


class SampleStoreDescription(pydantic.BaseModel):
    experiments: Annotated[
        list[SourceExperimentDescription],
        pydantic.Field(
            default_factory=list,
            description="A list describing the experiments in the source",
        ),
    ]
    generatorIdentifier: Annotated[
        str | None,
        pydantic.Field(description="The id of the entity generator"),
    ] = None

    @property
    def catalog(self) -> ExperimentCatalog:
        experiments = {}
        for desc in self.experiments:
            # Use the experiment property from ExperimentDescription
            experiment = desc.experiment
            experiments[experiment.identifier] = experiment

        return ExperimentCatalog(
            experiments=experiments, catalogIdentifier=self.generatorIdentifier
        )

    @property
    def experimentDescriptionMap(self) -> dict[str, SourceExperimentDescription]:
        return {e.experimentIdentifier: e for e in self.experiments}

    @property
    def observedProperties(self) -> list["ObservedProperty"]:
        """Return all observed properties defined by the receiver"""

        observedProperties: list[ObservedProperty] = []
        for e in self.catalog.experiments:
            observedProperties.extend(e.observedProperties)

        return observedProperties

    @property
    def constitutiveProperties(self) -> list[ConstitutiveProperty]:
        """Return constitutive property descriptors from all experiment descriptions"""
        from orchestrator.schema.property import ConstitutivePropertyDescriptor

        # Collect all unique constitutive property identifiers from experiment descriptions
        property_identifiers = set()
        for exp_desc in self.experiments:
            property_identifiers.update(exp_desc.constitutivePropertyMap.keys())

        return [
            ConstitutivePropertyDescriptor(identifier=prop_id)
            for prop_id in sorted(property_identifiers)
        ]

    @property
    def source_observed_property_identifiers(self) -> list[str]:
        """Returns all source column names for observed properties from all experiment descriptions"""
        identifiers = []
        for exp_desc in self.experiments:
            identifiers.extend(exp_desc.source_observed_property_identifiers)
        return identifiers

    @property
    def source_constitutive_property_identifiers(self) -> list[str]:
        """Returns all source column names for constitutive properties from all experiment descriptions"""
        identifiers = set()
        for exp_desc in self.experiments:
            identifiers.update(exp_desc.source_constitutive_property_identifiers)
        return sorted(identifiers)


class FailedToDecodeStoredEntityError(Exception):
    def __init__(
        self, entity_identifier: str, entity_representation: dict, cause: Exception
    ) -> None:
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
    ) -> None:
        self.entity_identifier = entity_identifier
        self.result_representation = result_representation
        self.cause = cause
        super().__init__(
            f"Unable to decode a measurement result for entity {entity_identifier}.\n\n"
            f"Result representation was: {result_representation}.\n\n"
            f"Error was: {cause}"
        )
