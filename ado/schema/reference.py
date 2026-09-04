# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from ado.schema.property import (
    ConstitutiveProperty,
)
from ado.schema.property_value import (
    ConstitutivePropertyValue,
)
from ado.utilities.pydantic import (
    StrictSemVerStr,
    semver_major,
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
    experimentVersion: Annotated[
        StrictSemVerStr | None,
        pydantic.Field(
            description=(
                "Algorithm version of the referenced experiment (strict SemVer "
                "MAJOR.MINOR.PATCH). When set, memoisation keys and equality checks "
                "encode only the MAJOR component so that minor/patch changes do not "
                "invalidate cached results."
            ),
        ),
    ] = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("experimentIdentifier")
    @classmethod
    def experiment_identifier_must_not_contain_at(cls, value: str) -> str:
        """Reject version suffixes embedded in experimentIdentifier."""
        if "@" in value:
            raise ValueError(
                "experimentIdentifier must not contain '@'. "
                "Use experimentVersion for algorithm versioning."
            )
        return value

    @property
    def major_version_experiment_identifier(self) -> str:
        """Return the major version experiment identifier.

        For versioned references this is ``'{base_name}@v{MAJOR}'``, e.g.
        ``'solve_mip@v1'``.  For unversioned references this is identical to
        :attr:`experimentIdentifier`.

        Returns:
            major version experiment identifier string.
        """
        if self.experimentVersion is not None:
            return (
                f"{self.experimentIdentifier}@v{semver_major(self.experimentVersion)}"
            )
        return self.experimentIdentifier

    @property
    def fully_qualified_experiment_identifier(self) -> str:
        """Return the fully-qualified experiment identifier encoding the exact version.

        For versioned references this is ``'{base_name}@{version}'``, e.g.
        ``'solve_mip@1.0.3'``.  For unversioned references this is identical to
        :attr:`experimentIdentifier`.

        Returns:
            Fully-qualified experiment identifier string.
        """
        if self.experimentVersion is not None:
            return f"{self.experimentIdentifier}@{self.experimentVersion}"
        return self.experimentIdentifier

    def __str__(self) -> str:
        return reference_string_from_fields(
            self.actuatorIdentifier,
            self.fully_qualified_parameterized_experiment_identifier,
        )

    def __repr__(self) -> str:
        return reference_string_from_fields(
            self.actuatorIdentifier,
            self.fully_qualified_parameterized_experiment_identifier,
        )

    def __eq__(self, other: object) -> bool:  # noqa: ANN401
        """Two references are equal when they have the same major_version_parameterized_experiment_identifier.

        Equality is based on the major version parameterized identifier, so references
        with the same base name, same major version, and same parameterization are
        equal regardless of minor/patch version differences.

        Note: when the references have no parameterization this is equivalent to
        comparing the major version experiment identifier.

        Returns:
            True if both references are for the same actuator and have the same
            parameterized experiment identifier.
        """
        retval = False
        if isinstance(other, ExperimentReference):
            retval = (self.actuatorIdentifier == other.actuatorIdentifier) and (
                self.major_version_parameterized_experiment_identifier
                == other.major_version_parameterized_experiment_identifier
            )

        return retval

    def __hash__(self) -> int:
        return hash(
            (
                self.actuatorIdentifier,
                self.major_version_parameterized_experiment_identifier,
            )
        )

    def validate_parameterization(self) -> None:
        """Validate the parameterization of this reference against the actuator catalog.

        Raises:
            ValueError: If the referenced experiment cannot be found or the
                parameterization is invalid.
        """
        from ado.modules.actuators.errors import UnknownExperimentError
        from ado.modules.actuators.registry import (
            ActuatorRegistry,
        )

        if self.parameterization is None:
            return

        try:
            experiment = ActuatorRegistry.globalRegistry().experimentForReference(
                ExperimentReference(
                    experimentIdentifier=self.experimentIdentifier,
                    actuatorIdentifier=self.actuatorIdentifier,
                    experimentVersion=self.experimentVersion,
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
    def major_version_parameterized_experiment_identifier(self) -> str:
        """Return the major version parameterized experiment identifier.

        Uses :attr:`major_version_experiment_identifier` as the prefix so that the
        memoisation key encodes the major algorithm version.

        * No version, no params: ``'solve_mip'`` (backward-compatible)
        * Version ``1.0.0``, no params: ``'solve_mip@v1'``
        * No version, with params: ``'solve_mip-time_limit_s.3600'`` (backward-compatible)
        * Version ``1.0.0``, with params: ``'solve_mip@v1-time_limit_s.3600'``

        Returns:
            major version parameterized identifier string.
        """
        return (
            identifier_for_parameterized_experiment(
                self.major_version_experiment_identifier, self.parameterization
            )
            if self.parameterization
            else self.major_version_experiment_identifier
        )

    @property
    def fully_qualified_parameterized_experiment_identifier(self) -> str:
        """Return the fully-qualified parameterized experiment identifier.

        Uses :attr:`fully_qualified_experiment_identifier` as the prefix.  This
        form is used by :meth:`__str__` for storage and display.

        Returns:
            Fully-qualified parameterized identifier string.
        """
        return (
            identifier_for_parameterized_experiment(
                self.fully_qualified_experiment_identifier, self.parameterization
            )
            if self.parameterization
            else self.fully_qualified_experiment_identifier
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
