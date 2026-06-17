# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import re
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.schema.property import (
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from orchestrator.schema.property_value import (
    ConstitutivePropertyValue,
)
from orchestrator.utilities.pydantic import StrictSemVerStr, semver_major

_FQ_VERSION_WITH_PARAMS_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(.*))?$"
)
# Split parameterization segments on '-' only before the next prop.value pair,
# not on '-' that begins a negative numeric value (e.g. test_opt2.-1).
_PARAMETERIZATION_SEGMENT_SPLIT = re.compile(r"-(?=[^.-][^.]*\.)")


def reference_string_from_fields(
    actuator_identifier: str, experiment_identifier: str
) -> str:
    """This method defines the identifier string used by ExperimentReference and Experiment"""

    return f"{actuator_identifier}.{experiment_identifier}"


def _parameterization_from_suffix(
    parameterization_suffix: str,
) -> list[ConstitutivePropertyValue]:
    """Parse a ``prop.val-prop2.val2`` suffix into property values."""
    parameterization: list[ConstitutivePropertyValue] = []
    segments = _PARAMETERIZATION_SEGMENT_SPLIT.split(parameterization_suffix)
    for segment in segments:
        if "." not in segment:
            raise ValueError(
                f"Invalid parameterization segment {segment!r} in experiment reference string. "
                "Expected 'propertyIdentifier.value'."
            )
        property_identifier, value_str = segment.split(".", maxsplit=1)
        parameterization.append(
            ConstitutivePropertyValue(
                property=ConstitutivePropertyDescriptor(identifier=property_identifier),
                value=value_str,
            )
        )
    return parameterization


def _parse_experiment_part_from_string(
    experiment_part: str,
) -> tuple[str, StrictSemVerStr | None, list[ConstitutivePropertyValue] | None]:
    """Parse the experiment portion of a reference string.

    Args:
        experiment_part: The substring after ``actuatorIdentifier.`` in a
            reference string representation.

    Returns:
        A tuple of base experiment identifier, optional strict SemVer version,
        and optional parameterization values.
    """
    if "@" in experiment_part:
        base_identifier, version_and_params = experiment_part.split("@", maxsplit=1)
        version_match = _FQ_VERSION_WITH_PARAMS_PATTERN.match(version_and_params)
        if version_match is not None:
            version: StrictSemVerStr = (
                f"{version_match.group(1)}.{version_match.group(2)}.{version_match.group(3)}"
            )
            parameterization_suffix = version_match.group(4)
            parameterization = (
                _parameterization_from_suffix(parameterization_suffix)
                if parameterization_suffix
                else None
            )
            return base_identifier, version, parameterization
        raise ValueError(
            f"Cannot parse version suffix in {experiment_part!r}. "
            "Version must be strict SemVer MAJOR.MINOR.PATCH (e.g. @1.0.0). "
            "Legacy forms like @v1 are not supported."
        )

    if "-" in experiment_part:
        base_identifier, parameterization_suffix = experiment_part.split(
            "-", maxsplit=1
        )
        if (
            parameterization_suffix
            and "." in _PARAMETERIZATION_SEGMENT_SPLIT.split(parameterization_suffix)[0]
        ):
            return (
                base_identifier,
                None,
                _parameterization_from_suffix(parameterization_suffix),
            )

    return experiment_part, None, None


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
            default=None,
            description=(
                "Algorithm version of the referenced experiment (strict SemVer "
                "MAJOR.MINOR.PATCH). When set, memoisation keys and equality checks "
                "encode only the MAJOR component so that minor/patch changes do not "
                "invalidate cached results."
            ),
        ),
    ]

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
    def semantic_experiment_identifier(self) -> str:
        """Return the semantic experiment identifier encoding the major version.

        For versioned references this is ``'{base_name}@v{MAJOR}'``, e.g.
        ``'solve_mip@v1'``.  For unversioned references this is identical to
        :attr:`experimentIdentifier`.

        Returns:
            Semantic experiment identifier string.
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

    @classmethod
    def referenceFromString(cls, stringRepresentation: str) -> "ExperimentReference":
        """Convert a string representation into an ExperimentReference.

        Parses strings of the form ``'{actuatorId}.{experimentId}'``, where
        ``experimentId`` may include an ``@MAJOR.MINOR.PATCH`` suffix and
        optional parameterization produced by :meth:`__str__` (e.g.
        ``'my_actuator.solve_mip@1.0.3'`` or
        ``'my_actuator.solve_mip@1.0.3-timeout.120'``).

        When no ``@MAJOR.MINOR.PATCH`` suffix is present, ``experimentVersion``
        is set to ``None``.

        The ``actuatorId`` must not contain periods.

        Args:
            stringRepresentation: The string to parse.

        Returns:
            A new ExperimentReference.

        Raises:
            ValueError: If the string contains no period separator.
        """
        try:
            actuator_identifier, experiment_part = stringRepresentation.split(
                ".", maxsplit=1
            )
        except Exception as error:
            raise ValueError(
                f"String, {stringRepresentation} is not a valid representation of an ExperimentReference. "
                f"At least one '.' is required to separate actuator id from experiment id. "
                f"If actuator id contains a period this method will not be able to parse the id from the reference string representation"
                f"Underlying error: {error}"
            ) from error

        (
            experiment_identifier,
            experiment_version,
            parameterization,
        ) = _parse_experiment_part_from_string(experiment_part)

        return cls(
            experimentIdentifier=experiment_identifier,
            actuatorIdentifier=actuator_identifier,
            experimentVersion=experiment_version,
            parameterization=parameterization,
        )

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
        """Two references are equal when they have the same parameterizedExperimentIdentifier.

        Equality is based on the semantic parameterized identifier, so references
        with the same base name, same major version, and same parameterization are
        equal regardless of minor/patch version differences.

        Note: when the references have no parameterization this is equivalent to
        comparing the semantic experiment identifier.

        Returns:
            True if both references are for the same actuator and have the same
            parameterized experiment identifier.
        """
        retval = False
        if isinstance(other, ExperimentReference):
            retval = (self.actuatorIdentifier == other.actuatorIdentifier) and (
                self.parameterizedExperimentIdentifier
                == other.parameterizedExperimentIdentifier
            )

        return retval

    def __hash__(self) -> int:
        return hash((self.actuatorIdentifier, self.parameterizedExperimentIdentifier))

    def validate_parameterization(self) -> None:
        """Validate the parameterization of this reference against the actuator catalog.

        Raises:
            ValueError: If the referenced experiment cannot be found or the
                parameterization is invalid.
        """
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
    def parameterizedExperimentIdentifier(self) -> str:
        """Return the semantic parameterized experiment identifier.

        Uses :attr:`semantic_experiment_identifier` as the prefix so that the
        memoisation key encodes the major algorithm version.

        * No version, no params: ``'solve_mip'`` (backward-compatible)
        * Version ``1.0.0``, no params: ``'solve_mip@v1'``
        * No version, with params: ``'solve_mip-time_limit_s.3600'`` (backward-compatible)
        * Version ``1.0.0``, with params: ``'solve_mip@v1-time_limit_s.3600'``

        Returns:
            Semantic parameterized identifier string.
        """
        return (
            identifier_for_parameterized_experiment(
                self.semantic_experiment_identifier, self.parameterization
            )
            if self.parameterization
            else self.semantic_experiment_identifier
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
