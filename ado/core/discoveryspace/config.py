# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from ado.core.metadata import ConfigurationMetadata
from ado.schema.measurementspace import MeasurementSpaceConfiguration
from ado.schema.property import ConstitutiveProperty
from ado.schema.reference import ExperimentReference


class SpaceHierarchy(enum.Enum):
    SUB_SPACE = "⊂ SUBSPACE"
    SUPER_SPACE = "⊃ SUPERSPACE"
    EQUAL_SPACE = "= IDENTICAL"
    OVERLAPPING = "∩ OVERLAPPING"
    DISJOINT = "∅ DISJOINT"
    UNDEFINED = "? UNDEFINED"


class VersionMatch(enum.Enum):
    IDENTICAL = "= IDENTICAL"
    MAJOR_ONLY = "≈ MAJOR_ONLY"


class ParameterizationMatch(enum.Enum):
    IDENTICAL = "= IDENTICAL"
    DIFFERENT = "≠ DIFFERENT"
    NOT_APPLICABLE = "N/A"


class MeasurementSpaceComparisonResult(pydantic.BaseModel):
    measurement_space_match: Annotated[
        SpaceHierarchy,
        pydantic.Field(
            description="How the sets of experiments relate",
        ),
    ]
    version_match: Annotated[
        VersionMatch | None,
        pydantic.Field(
            description="Version match for the overlapping experiments",
        ),
    ] = None
    parameterization_match: Annotated[
        ParameterizationMatch | None,
        pydantic.Field(
            description="Parameterization match for the overlapping experiments",
        ),
    ] = None
    model_config = ConfigDict(extra="forbid")


def ms_config_type_discriminator(
    ms_config: list | dict | MeasurementSpaceConfiguration,
) -> str:

    if isinstance(ms_config, list):
        return "ExperimentReferenceList"
    if isinstance(ms_config, MeasurementSpaceConfiguration):
        return "MeasurementSpaceConfiguration"
    if isinstance(ms_config, dict) and ms_config.get("experiments"):
        return "MeasurementSpaceConfiguration"

    raise ValueError(
        f"Unable to determine type measurement space configuration type of data, {ms_config}"
    )


MeasurementSpaceConfigurationType = typing.Annotated[
    typing.Annotated[
        MeasurementSpaceConfiguration, pydantic.Tag("MeasurementSpaceConfiguration")
    ]
    | typing.Annotated[
        list[ExperimentReference], pydantic.Tag("ExperimentReferenceList")
    ],
    pydantic.Discriminator(ms_config_type_discriminator),
]


class DiscoverySpaceProperties(pydantic.BaseModel):
    stochastic: Annotated[
        bool,
        pydantic.Field(
            description="If true the values for properties are drawn from a distribution."
            "Use this field when this is known to be true for all properties"
        ),
    ] = False


class DiscoverySpaceConfiguration(pydantic.BaseModel):
    sampleStoreIdentifier: Annotated[
        str,
        pydantic.Field(
            description="The id of the sample store to use.",
            coerce_numbers_to_str=True,
        ),
    ] = "default"
    entitySpace: Annotated[
        list[ConstitutiveProperty] | None,
        pydantic.Field(
            description="Describes how entities can be generated in this space"
        ),
    ] = None
    experiments: Annotated[
        MeasurementSpaceConfigurationType | None,
        pydantic.Field(description="Defines the measurement space"),
    ] = None
    metadata: Annotated[
        ConfigurationMetadata,
        pydantic.Field(
            description="Metadata about the configuration including optional name, description, "
            "labels for filtering, and any additional custom fields"
        ),
    ] = ConfigurationMetadata()
    model_config = ConfigDict(extra="forbid")

    @pydantic.field_validator("entitySpace", mode="before")
    @classmethod
    def strip_binary_variable_types_data(
        cls, values: dict | list[ConstitutiveProperty]
    ) -> dict:
        from pydantic import TypeAdapter

        from ado.core.resources import (
            CoreResourceKinds,
            warn_deprecated_resource_model_in_use,
        )
        from ado.schema.domain import VariableTypeEnum

        if values is None:
            return values

        if isinstance(values, list) and all(
            isinstance(item, ConstitutiveProperty) for item in values
        ):
            values = TypeAdapter(list[ConstitutiveProperty]).dump_python(values)

        property_domain_key = "propertyDomain"
        variable_type_key = "variableType"
        for property in values:
            property_domain = property.get(property_domain_key)
            if property_domain_key not in property or not property_domain:
                continue

            if (
                variable_type_key not in property_domain
                or property_domain.get(variable_type_key)
                != VariableTypeEnum.BINARY_VARIABLE_TYPE.value
            ):
                continue

            keys_to_remove = {
                "values",
                "interval",
                "domainRange",
                "probabilityFunction",
            }
            overlapping_keys = set(property[property_domain_key].keys()).difference(
                keys_to_remove
            )

            # If the propertyDomain has any of the keys above, we rewrite it completely
            # to just contain BINARY_VARIABLE_TYPE instead of popping each key.
            if overlapping_keys:
                property[property_domain_key] = {
                    variable_type_key: VariableTypeEnum.BINARY_VARIABLE_TYPE.value
                }
                warn_deprecated_resource_model_in_use(
                    affected_resource=CoreResourceKinds.DISCOVERYSPACE,
                    deprecated_from_ado_version="1.6.0",
                    removed_from_ado_version="2.0.0",
                )

        return values

    def convert_experiments_to_reference_list(self) -> "DiscoverySpaceConfiguration":
        """Returns a copy where the experiments field is a list of experiment references"""

        if not isinstance(self.experiments, MeasurementSpaceConfiguration):
            return self.model_copy()

        return self.model_copy(
            update={
                "experiments": [
                    experiment.reference for experiment in self.experiments.experiments
                ]
            }
        )

    def convert_experiments_to_measurement_space_config(
        self,
    ) -> "DiscoverySpaceConfiguration":
        """Returns a copy where the experiments field is a MeasurementSpaceConfiguration instance"""

        from ado.modules.actuators.registry import ActuatorRegistry

        if isinstance(self.experiments, MeasurementSpaceConfiguration):
            return self.model_copy()

        gr = ActuatorRegistry.globalRegistry()

        experiments = [
            gr.experimentForReference(
                ref, match_on="fully_qualified_version", resolve=True
            )
            for ref in self.experiments
        ]

        return self.model_copy(
            update={
                "experiments": MeasurementSpaceConfiguration(experiments=experiments)
            }
        )

    def is_sub_space(self, reference_space: "DiscoverySpaceConfiguration") -> bool:

        if not self.entitySpace:
            raise ValueError("The target entity space was empty")

        if not reference_space.entitySpace:
            raise ValueError("The reference entity space was empty")

        # We create maps from the constitutive properties to make it
        # easier to do our comparisons
        target_space_constitutive_property_map = {
            constitutive_property.identifier: constitutive_property.propertyDomain
            for constitutive_property in self.entitySpace
        }
        reference_space_constitutive_property_map = {
            constitutive_property.identifier: constitutive_property.propertyDomain
            for constitutive_property in reference_space.entitySpace
        }

        # To be a subspace, the target space must not have more properties
        # than the reference
        if len(target_space_constitutive_property_map.keys()) > len(
            reference_space_constitutive_property_map.keys()
        ):
            return False

        for identifier, domain in target_space_constitutive_property_map.items():
            # The target space has a property that the reference space does not have
            if identifier not in reference_space_constitutive_property_map:
                return False

            if not domain.isSubDomain(
                reference_space_constitutive_property_map[identifier]
            ):
                return False

        return True

    def overlaps(self, other: "DiscoverySpaceConfiguration") -> bool:
        """Returns True if the two spaces share at least one point.

        Two spaces overlap when, for every property identifier present in
        **both** spaces, the corresponding domains overlap.  If no property
        identifiers are shared the spaces are considered disjoint.

        Parameters:
            other: The other DiscoverySpaceConfiguration to compare against
        Returns:
            True if the spaces overlap
        """
        self_property_domain_by_id = {
            cp.identifier: cp.propertyDomain for cp in (self.entitySpace or [])
        }
        other_property_domain_by_id = {
            cp.identifier: cp.propertyDomain for cp in (other.entitySpace or [])
        }

        shared_property_ids = set(self_property_domain_by_id).intersection(
            other_property_domain_by_id
        )
        if not shared_property_ids:
            return False

        return all(
            self_property_domain_by_id[property_id].overlaps(
                other_property_domain_by_id[property_id]
            )
            for property_id in shared_property_ids
        )

    def compare_entity_space(
        self, reference_space: "DiscoverySpaceConfiguration"
    ) -> SpaceHierarchy:
        """Compares the entity space hierarchy between self and reference_space.

        Parameters:
            reference_space: The reference DiscoverySpaceConfiguration

        Returns:
            A SpaceHierarchy enum value describing the relationship
        """
        try:
            is_sub_space = self.is_sub_space(reference_space)
            is_super_space = reference_space.is_sub_space(self)
        except ValueError:
            return SpaceHierarchy.UNDEFINED

        if is_sub_space and is_super_space:
            return SpaceHierarchy.EQUAL_SPACE
        if is_sub_space:
            return SpaceHierarchy.SUB_SPACE
        if is_super_space:
            return SpaceHierarchy.SUPER_SPACE

        # Neither is a sub-space — check for overlap
        if self.overlaps(reference_space):
            return SpaceHierarchy.OVERLAPPING
        return SpaceHierarchy.DISJOINT

    def compare_measurement_space(
        self, reference_space: "DiscoverySpaceConfiguration"
    ) -> MeasurementSpaceComparisonResult:
        """Compares the measurement space between self (candidate) and reference_space.

        Parameters:
            reference_space: The reference DiscoverySpaceConfiguration.

        Returns:
            A MeasurementSpaceComparisonResult describing how the measurement spaces relate,
            along with version and parameterization matching across shared experiments.
        """
        candidate_converted = self.convert_experiments_to_reference_list()
        reference_converted = reference_space.convert_experiments_to_reference_list()

        candidate_experiment_references = candidate_converted.experiments or []
        reference_experiment_references = reference_converted.experiments or []

        candidate_experiments_by_key = {
            (
                experiment_reference.actuatorIdentifier,
                experiment_reference.major_version_experiment_identifier,
            ): experiment_reference
            for experiment_reference in candidate_experiment_references
        }
        reference_experiments_by_key = {
            (
                experiment_reference.actuatorIdentifier,
                experiment_reference.major_version_experiment_identifier,
            ): experiment_reference
            for experiment_reference in reference_experiment_references
        }

        candidate_keys = set(candidate_experiments_by_key.keys())
        reference_keys = set(reference_experiments_by_key.keys())

        if not candidate_keys or not reference_keys:
            return MeasurementSpaceComparisonResult(
                measurement_space_match=SpaceHierarchy.DISJOINT,
            )

        if candidate_keys.isdisjoint(reference_keys):
            return MeasurementSpaceComparisonResult(
                measurement_space_match=SpaceHierarchy.DISJOINT,
            )

        if candidate_keys == reference_keys:
            measurement_space_match = SpaceHierarchy.EQUAL_SPACE
        elif candidate_keys.issuperset(reference_keys):
            measurement_space_match = SpaceHierarchy.SUPER_SPACE
        elif candidate_keys.issubset(reference_keys):
            measurement_space_match = SpaceHierarchy.SUB_SPACE
        else:
            measurement_space_match = SpaceHierarchy.OVERLAPPING

        # Compute version match across all shared experiment pairs
        # Report least powerful outcome: one MAJOR_ONLY makes it MAJOR_ONLY
        common_keys = candidate_keys.intersection(reference_keys)
        has_major_only = False
        for key in common_keys:
            candidate_experiment = candidate_experiments_by_key[key]
            reference_experiment = reference_experiments_by_key[key]
            if (
                candidate_experiment.fully_qualified_experiment_identifier
                != reference_experiment.fully_qualified_experiment_identifier
            ):
                has_major_only = True
                break

        version_match = (
            VersionMatch.MAJOR_ONLY if has_major_only else VersionMatch.IDENTICAL
        )

        # Compute parameterization match across all shared experiment pairs
        # Report least powerful outcome:
        # If any pair is DIFFERENT -> DIFFERENT
        # Else if any pair is IDENTICAL -> IDENTICAL
        # Else (all N/A) -> NOT_APPLICABLE
        has_different = False
        has_identical = False
        for key in common_keys:
            candidate_experiment = candidate_experiments_by_key[key]
            reference_experiment = reference_experiments_by_key[key]
            if (
                not candidate_experiment.parameterization
                and not reference_experiment.parameterization
            ):
                continue

            if (
                candidate_experiment.major_version_parameterized_experiment_identifier
                == reference_experiment.major_version_parameterized_experiment_identifier
            ):
                has_identical = True
            else:
                has_different = True
                break

        if has_different:
            parameterization_match = ParameterizationMatch.DIFFERENT
        elif has_identical:
            parameterization_match = ParameterizationMatch.IDENTICAL
        else:
            parameterization_match = ParameterizationMatch.NOT_APPLICABLE

        return MeasurementSpaceComparisonResult(
            measurement_space_match=measurement_space_match,
            version_match=version_match,
            parameterization_match=parameterization_match,
        )

    def validate_entity_space_against_measurement_space(self) -> None:
        """Validates the entity space against the measurement space.

        Creates a MeasurementSpace instance and EntitySpaceRepresentation from the
        configuration and uses MeasurementSpace.checkEntitySpaceCompatible to validate
        compatibility.

        Raises:
            ValueError: If validation fails. The error message will describe the
                incompatibility between the entity space and measurement space, or
                if experiments cannot be resolved (e.g., experiments from sample store
                catalogs are not available in dry-run mode).
            UnknownExperimentError: If experiments cannot be resolved
            (e.g., actuator is not installed).
        """

        from ado.schema.entityspace import EntitySpaceRepresentation
        from ado.schema.measurementspace import MeasurementSpace

        # Skip validation if either entity space or experiments are not defined
        if self.entitySpace is None or self.experiments is None:
            return

        # Create EntitySpaceRepresentation from the entity space configuration
        entity_space = EntitySpaceRepresentation.representationFromConfiguration(
            self.entitySpace
        )

        if isinstance(self.experiments, MeasurementSpaceConfiguration):
            # If we have full MeasurementSpaceConfiguration we can initialize directly
            measurement_space = MeasurementSpace(configuration=self.experiments)
        else:
            # Otherwise we have to use registry to reconstruct the experiments
            # This will use the actuator registry to resolve experiment references
            measurement_space = MeasurementSpace.measurementSpaceFromSelection(
                selectedExperiments=self.experiments,
                experimentCatalogs=None,  # In dry-run, we don't have sample store catalogs
            )

        # Validate compatibility - this will raise ValueError if validation fails
        measurement_space.checkEntitySpaceCompatible(entitySpace=entity_space)


class EntityFilter(enum.Enum):
    SAMPLED = "sampled"  # Only entities sampled by operations on the space
    MATCHING = "matching"  # All entities in the source that match the space
    ALL = "all"  # All entities in the source
