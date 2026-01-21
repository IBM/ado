# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.schema.property import ConstitutiveProperty
from orchestrator.schema.property_value import (
    ConstitutivePropertyValue,
)


def reference_string_from_fields(
    actuator_identifier: str, experiment_identifier: str
) -> str:
    """This method defines the identifier string used by ExperimentReference and Experiment"""

    return f"{actuator_identifier}.{experiment_identifier}"


class ExperimentReference(pydantic.BaseModel):
    experimentIdentifier: Annotated[
        str,
        pydantic.Field(
            description="The identifier of an experiment in an actuator experiment catalog"
        ),
    ]
    actuatorIdentifier: Annotated[
        str,
        pydantic.Field(
            description="The identifier of the actuator that supplies the experiment"
        ),
    ]
    parameterization: Annotated[
        list[ConstitutivePropertyValue] | None,
        pydantic.Field(
            description="A list of values for optional properties of the experiment"
        ),
    ] = None

    model_config = ConfigDict(frozen=True)

    @classmethod
    def referenceFromString(cls, stringRepresentation: str) -> "ExperimentReference":
        """Convert a string representation of a reference into a ExperimentReference instance, if possible

        This method relied on the actuator id not containing any periods as this is the separator
        in the string representation of an ExperimentReference between the actuator and the experiment

        Raises ValueError if the string contains no periods"""

        try:
            actuatorIdentifier, experimentIdentifier = stringRepresentation.split(
                ".", maxsplit=1
            )
        except Exception as error:
            raise ValueError(
                f"String, {stringRepresentation} is not a valid representation of an ExperimentReference. "
                f"At least one '.' is required to separate actuator id from experiment id. "
                f"If actuator id contains a period this method will not be able to parse the id from the reference string representation"
                f"Underlying error: {error}"
            ) from error
        else:
            return cls(
                experimentIdentifier=experimentIdentifier,
                actuatorIdentifier=actuatorIdentifier,
            )

    def compareWithoutParameterization(self, other: "ExperimentReference") -> bool:
        """Compares to other using actuator id and base experiment id, no parameterization

        If this method returns true if the two references refer to the same experiment in the same actuator,
        although they may be parameterized differently"""

        return (
            self.actuatorIdentifier == other.actuatorIdentifier
            and self.experimentIdentifier == other.experimentIdentifier
        )

    def __str__(self) -> str:
        return reference_string_from_fields(
            self.actuatorIdentifier, self.parameterizedExperimentIdentifier
        )

    def __repr__(self) -> str:
        return reference_string_from_fields(
            self.actuatorIdentifier, self.parameterizedExperimentIdentifier
        )

    def __eq__(self, other: object) -> bool:  # noqa: ANN401
        """Two references, refer to same experiment if they have same parameterizedExperimentIdentifier

                Note: when the references have no parameterization this is equivalent to comparing the experimentIdentifier
        ="""

        retval = False
        if isinstance(other, ExperimentReference):
            retval = (self.actuatorIdentifier == other.actuatorIdentifier) and (
                self.parameterizedExperimentIdentifier
                == other.parameterizedExperimentIdentifier
            )

        return retval

    def __hash__(self) -> int:
        return hash(str(self))

    def validate_parameterization(self) -> None:

        from orchestrator.modules.actuators.registry import (
            ActuatorRegistry,
            UnknownExperimentError,
        )

        if self.parameterization is None:
            return

        try:
            experiment = ActuatorRegistry.globalRegistry().experimentForReference(
                ExperimentReference(
                    experimentIdentifier=self.experimentIdentifier,
                    actuatorIdentifier=self.actuatorIdentifier,
                )
            )
        except UnknownExperimentError as error:
            raise ValueError(
                "Failed validating parameterization. "
                f"Cannot find experiment {self.experimentIdentifier} from actuator {self.actuatorIdentifier} in catalog"
            ) from error
        else:
            if not experiment.optionalProperties and self.parameterization:
                raise ValueError(
                    f"Experiment reference {self} specifies custom parameterization "
                    f"but the referenced experiment has no parameterizable properties."
                )

            check_parameterization_validity(
                parameterizableProperties=experiment.optionalProperties,
                customParameterization=self.parameterization,
                defaultParameterization=experiment.defaultParameterization,
            )

    @property
    def parameterizedExperimentIdentifier(self) -> str:
        """
        If a parameterization is given, this will be the identifier of the
        parameterized version of the experiment, otherwise it is
        identical to experimentIdentifier.
        """
        return (
            identifier_for_parameterized_experiment(
                self.experimentIdentifier, self.parameterization
            )
            if self.parameterization
            else self.experimentIdentifier
        )


def identifier_for_parameterized_experiment(
    identifier: str, parameterization: list[ConstitutivePropertyValue]
) -> str:

    # Check the parameterized experiments id is as expected.
    # We construct it here as it's expected to be done
    pstr = "-".join([f"{v.property.identifier}.{v.value}" for v in parameterization])

    return f"{identifier}-{pstr}"


def check_parameterization_validity(
    parameterizableProperties: list[ConstitutiveProperty],
    customParameterization: typing.Iterable[ConstitutivePropertyValue],
    defaultParameterization: list[ConstitutivePropertyValue] | None = None,
) -> None:
    """Checks if values are a valid parameterization of properties"""

    if parameterizableProperties is None:
        raise ValueError(
            "Passed None for parameterizableProperties to check_parameterization_validity"
        )

    if customParameterization is None:
        raise ValueError(
            "Passed None for customParameterization to check_parameterization_validity"
        )

    # Check all parameterized properties are in properties
    mapping = {c.identifier: c for c in parameterizableProperties}
    hasNoProperty = [
        v for v in customParameterization if mapping.get(v.property.identifier) is None
    ]
    if len(hasNoProperty) > 0:
        raise ValueError(
            f"parameterized properties not in optionalProperties list. Missing: {[v.property.identifier for v in hasNoProperty]}"
        )

    # Check there are no duplicate properties
    propertiesParameterized = [v.property for v in customParameterization]
    if len({p.identifier for p in propertiesParameterized}) != len(
        [p.identifier for p in propertiesParameterized]
    ):
        raise ValueError(
            "The parameterization contains multiple values for same property"
        )

    # Check all values are in domain
    for v in customParameterization:
        prop = mapping[v.property.identifier]
        if not prop.propertyDomain.valueInDomain(v.value):
            raise ValueError(
                f"Parameterized value, {v.value}, for property {prop.identifier} is not in the properties domain {prop.propertyDomain}"
            )

    if defaultParameterization:
        defaultMapping = {v.property.identifier: v for v in defaultParameterization}
        # Check all values are different to the defaults
        for v in customParameterization:
            if v.value == defaultMapping[v.property.identifier].value:
                raise ValueError(
                    f"Custom parameterization for property {v.property.identifier} with value {v.value} has same value as default parameterization: {defaultMapping[v.property.identifier]}"
                )
