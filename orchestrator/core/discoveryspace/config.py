# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.schema.experiment import ParameterizedExperiment
from orchestrator.schema.measurementspace import MeasurementSpaceConfiguration
from orchestrator.schema.property import ConstitutiveProperty
from orchestrator.schema.reference import ExperimentReference


class SpaceHierarchy(enum.Enum):
    SUB_SPACE = "subspace"
    SUPER_SPACE = "superspace"
    EQUAL_SPACE = "equal"
    UNDEFINED = "undefined"


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

    def convert_experiments_to_reference_list(self) -> "DiscoverySpaceConfiguration":
        """Returns a copy where the experiments field is a list of experiment references"""

        if not isinstance(self.experiments, MeasurementSpaceConfiguration):
            return self.model_copy()

        return self.model_copy(
            update={
                "experiments": [
                    ExperimentReference(
                        experimentIdentifier=experiment.identifier,
                        actuatorIdentifier=experiment.actuatorIdentifier,
                        parameterization=(
                            experiment.parameterization
                            if isinstance(experiment, ParameterizedExperiment)
                            else None
                        ),
                    )
                    for experiment in self.experiments.experiments
                ]
            }
        )

    def convert_experiments_to_measurement_space_config(
        self,
    ) -> "DiscoverySpaceConfiguration":
        """Returns a copy where the experiments field is a MeasurementSpaceConfiguration instance"""

        from orchestrator.modules.actuators.registry import ActuatorRegistry

        if isinstance(self.experiments, MeasurementSpaceConfiguration):
            return self.model_copy()

        gr = ActuatorRegistry.globalRegistry()

        experiments = []
        for ref in self.experiments:
            if ref.parameterization:
                # We have to fetch the base experiment from the registry and parameterize it
                experiments.append(
                    ParameterizedExperiment(
                        parameterization=ref.parameterization,
                        **gr.experimentForReference(ref).model_dump(),
                    )
                )
            else:
                experiments.append(gr.experimentForReference(ref))

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

    def compare_space_hierarchy(
        self, reference_space: "DiscoverySpaceConfiguration"
    ) -> SpaceHierarchy:
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
        return SpaceHierarchy.UNDEFINED


class EntityFilter(enum.Enum):
    SAMPLED = "sampled"  # Only entities sampled by operations on the space
    MATCHING = "matching"  # All entities in the source that match the space
    ALL = "all"  # All entities in the source
