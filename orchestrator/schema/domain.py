# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import logging
import math
import typing
from typing import Annotated

import numpy as np
import pydantic
from pydantic import ConfigDict

if typing.TYPE_CHECKING:
    from rich.console import RenderableType


class VariableTypeEnum(str, enum.Enum):
    """Used to denote if the values of a property are discrete, continuous etc."""

    CONTINUOUS_VARIABLE_TYPE = (
        "CONTINUOUS_VARIABLE_TYPE"  # the value of the variable is continuous
    )
    DISCRETE_VARIABLE_TYPE = (
        "DISCRETE_VARIABLE_TYPE"  # the value of the variable is discrete
    )
    CATEGORICAL_VARIABLE_TYPE = (
        "CATEGORICAL_VARIABLE_TYPE"  # the value of the variable is a category label
    )
    OPEN_CATEGORICAL_VARIABLE_TYPE = "OPEN_CATEGORICAL_VARIABLE_TYPE"  # the value of the variable is a category label but all categories are not known in advance
    BINARY_VARIABLE_TYPE = "BINARY_VARIABLE_TYPE"  # the value of the variable is binary
    UNKNOWN_VARIABLE_TYPE = "UNKNOWN_VARIABLE_TYPE"  # the type of value of the variable is unknown/unspecified
    IDENTIFIER_VARIABLE_TYPE = "IDENTIFIER_VARIABLE_TYPE"  # the value is some type of, possible unique, identifier


class ProbabilityFunctionsEnum(str, enum.Enum):
    UNIFORM = "uniform"  # A uniform distribution
    NORMAL = "normal"  # A normal distribution
    # Can easily add more


def is_float_range(
    interval: float,
    domain_range: list[int | float],
) -> bool:
    "Returns True if an on interval or domain range is a float"

    return any(isinstance(x, float) for x in [interval, *domain_range])


def _internal_range_values(lower: float, upper: float, interval: float) -> list:
    """Returns the values in the half-open [lower,upper) range

    If all values are integers uses arange
    If one value is a float uses linspace and then removes the last value

    All values are rounded to 10 decimal places

    This function is required due to floating precision issues.
    The rounding deals with issues like 0.2+0.1 = 0.30000000000000004
    linspace delas with issue like arange(0.1,0.4,0.1) includes 0.4

    """

    if not is_float_range(interval=interval, domain_range=[lower, upper]):
        return list(np.arange(lower, upper, interval))
    num = int(np.floor((upper - lower) / interval)) + 1
    values = [lower + i * interval for i in range(num)]
    if values[-1] == upper:
        values = values[:-1]
    # values = np.linspace(lower, upper, num)[:-1]
    return list(np.round(values, 10))


def is_subdomain_of_unknown_domain(
    unknownDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if the testDomain is a subdomain of the unknownDomain
    Parameters:
        unknownDomain: A PropertyDomain with variableType UNKNOWN_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the testDomain is a subdomain of the unknownDomain
    """
    return True


def is_subdomain_of_continuous_domain(
    continuousDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if the testDomain is a subdomain of the continuousDomain
    Parameters:
        continuousDomain: A PropertyDomain with variableType CONTINUOUS_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the testDomain is a subdomain of the continuousDomain
    """
    if testDomain.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
        return continuousDomain.domainRange is None or (
            min(continuousDomain.domainRange) <= min(testDomain.domainRange)
            and max(continuousDomain.domainRange) >= max(testDomain.domainRange)
        )
    if testDomain.variableType in [
        VariableTypeEnum.UNKNOWN_VARIABLE_TYPE,
        VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
        VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
    ]:
        return False
    if testDomain.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
        if continuousDomain.domainRange is None:
            return True

        if testDomain.size == math.inf:
            return False

        return min(continuousDomain.domainRange) <= min(
            testDomain.domain_values
        ) and max(continuousDomain.domainRange) > max(testDomain.domain_values)

    # The only variable type left is BINARY
    # Check 0,1 is within our domainRange
    return continuousDomain.domainRange is None or (
        min(continuousDomain.domainRange) <= 0 and max(continuousDomain.domainRange) > 1
    )


def is_subdomain_of_discrete_domain(
    discreteDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if the testDomain is a subdomain of the discreteDomain
    Parameters:
        discreteDomain: A PropertyDomain with variableType DISCRETE_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the testDomain is a subdomain of the discreteDomain
    """
    if testDomain.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
        if discreteDomain.interval:
            if testDomain.interval:
                # We both have  an interval
                # Our interval must be divisible by domain interval
                if testDomain.interval % discreteDomain.interval == 0:
                    # Now we have to check the ranges
                    if testDomain.domainRange and discreteDomain.domainRange:
                        # Both have ranges - values must be subsets of each other
                        retval = set(testDomain.domain_values).issubset(
                            discreteDomain.domain_values
                        )
                    elif (
                        testDomain.domainRange and not discreteDomain.domainRange
                    ) or (not testDomain.domainRange and discreteDomain.domainRange):
                        # If we have a range and the other doesn't, it's a subdomain; if we don't have a range and the other does, it's not a subdomain
                        retval = (
                            testDomain.domainRange and not discreteDomain.domainRange
                        )
                    else:
                        # Neither have ranges
                        retval = True
                else:
                    retval = False
            else:
                # they have a domain range and interval we have values
                # convert their domain range to values
                retval = set(testDomain.domain_values).issubset(
                    discreteDomain.domain_values
                )
        else:
            retval = set(testDomain.domain_values).issubset(
                discreteDomain.domain_values
            )
        return retval
    if testDomain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
        return all(x in discreteDomain.domain_values for x in [0, 1])

    # All other domains are false (CONTINUOUS, OPEN_CATEGORICAL, UNKNOWN and CATEGORICAL
    return False


def is_subdomain_of_categorical_domain(
    categoricalDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if the testDomain is a subdomain of the categoricalDomain
    Parameters:
        categoricalDomain: A PropertyDomain with variableType CATEGORICAL_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the testDomain is a subdomain of the categoricalDomain
    """
    # Check against all members of VariableTypeEnum

    if testDomain.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
        return all(x in categoricalDomain.values for x in testDomain.values)
    if testDomain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
        return all(x in categoricalDomain.domain_values for x in [0, 1])
    if testDomain.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
        return testDomain.size != math.inf and all(
            x in categoricalDomain.domain_values for x in testDomain.domain_values
        )

    # All other domains are false: OPEN_CATEGORICAL, UNKNOWN, CONTINUOUS
    return False


def is_subdomain_of_binary_domain(
    binaryDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if the testDomain is a subdomain of the binaryDomain

    The cases where this returns True are
    - testDomain is a BINARY_VARIABLE_TYPE
    - testDomain is a DISCRETE_VARIABLE_TYPE or CATEGORICAL_VARIABLE_TYPE with size <= 2
    and all values in testDomain are in binaryDomain.domain_values

    Parameters:
        binaryDomain: A PropertyDomain with variableType BINARY_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the testDomain is a subdomain of the binaryDomain
    """
    if testDomain.variableType in [
        VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
    ]:
        if testDomain.size <= 2:
            return all(
                x in binaryDomain.domain_values for x in testDomain.domain_values
            )
        return False

    # All other domains are false: OPEN_CATEGORICAL, UNKNOWN, CONTINUOUS except BINARY
    return testDomain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE


def is_subdomain_of_open_categorical_domain(
    openCategoricalDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if the testDomain is a subdomain of the openCategoricalDomain

    The cases where this returns True are:
    - testDomain is an OPEN_CATEGORICAL_VARIABLE_TYPE, BINARY_VARIABLE_TYPE or CATEGORICAL_VARIABLE_TYPE
    - testDomain is a Discrete_VARIABLE_TYPE with size != math.inf

    Parameters:
        openCategoricalDomain: A PropertyDomain with variableType OPEN_CATEGORICAL_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the testDomain is a subdomain of the openCategoricalDomain
    """
    if testDomain.variableType in [
        VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
        VariableTypeEnum.BINARY_VARIABLE_TYPE,
        VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
    ]:
        return True
    if testDomain.variableType in [
        VariableTypeEnum.UNKNOWN_VARIABLE_TYPE,
        VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
    ]:
        return False

    # Only domain left is DISCRETE
    return testDomain.size != math.inf


class ProbabilityFunction(pydantic.BaseModel):
    identifier: Annotated[ProbabilityFunctionsEnum, pydantic.Field()] = (
        ProbabilityFunctionsEnum.UNIFORM
    )
    # Whatever parameters the probability function takes.
    # Should take range, interval, and categories
    parameters: Annotated[dict | None, pydantic.Field()] = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __eq__(self, other: object) -> bool:  # noqa: ANN401

        # They should both be ProbabilityFunctions
        if not isinstance(other, ProbabilityFunction):
            return False

        # Identifiers must match
        if self.identifier != other.identifier:
            logging.debug(
                f"Probability functions type is not the same {self.identifier, other.identifier}"
            )
            return False

        # If this instance doesn't have parameters, the other one
        # shouldn't have them either
        if not self.parameters:
            return not other.parameters

        # This instance has parameters. The other instance should
        # have the same parameter keys and values
        if not other.parameters or self.parameters.keys() != other.parameters.keys():
            logging.debug(
                f"The other probability function has a different number of parameters: {self.parameters, other.parameters}"
            )
            return False

        for key in self.parameters:
            if self.parameters[key] != other.parameters[key]:
                logging.debug(
                    f"The value of parameter {key} differs: {self.parameters[key], other.parameters[key]}"
                )
                return False

        return True


class PropertyDomain(pydantic.BaseModel):
    """Describes the domain of a property"""

    values: Annotated[
        list[typing.Any] | None,
        pydantic.Field(description="The values for a discrete or categorical domain"),
    ] = None
    interval: Annotated[
        int | float | None,
        pydantic.Field(
            description="The interval between discrete values variables. Do not set if values is set"
        ),
    ] = None  # Only makes sense for discrete variables.
    domainRange: Annotated[
        list[int | float] | None,
        pydantic.Field(
            description="The range of the domain for discrete or continuous variables. Inclusive of lower bound exclusive of upper bound. Calculated automatically if values is given.",
            validate_default=True,
            min_length=2,
            max_length=2,
            frozen=True,
        ),
    ] = None  # For discrete/continuous variables
    variableType: Annotated[VariableTypeEnum, pydantic.Field(validate_default=True)] = (
        VariableTypeEnum.UNKNOWN_VARIABLE_TYPE
    )
    probabilityFunction: Annotated[ProbabilityFunction, pydantic.Field()] = (
        ProbabilityFunction()
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __rich__(self) -> "RenderableType":
        """Render this property domain using rich."""
        from rich.console import Group
        from rich.text import Text

        from orchestrator.utilities.rich import get_rich_repr

        lines = [
            Text.assemble(("Type: ", "bold"), self.variableType),
        ]
        if self.values:
            lines.extend(
                [Text("Values:", style="bold", end=" "), get_rich_repr(self.values)]
            )
        if self.interval:
            lines.extend(
                [Text("Interval:", style="bold", end=" "), get_rich_repr(self.interval)]
            )
        if self.domainRange and self.variableType in [
            VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
            VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        ]:
            lines.extend(
                [Text("Range:", style="bold", end=" "), get_rich_repr(self.domainRange)]
            )

        return Group(*lines)

    @pydantic.field_validator("interval")
    def interval_requires_no_values(
        cls, interval: float | None, values: "pydantic.FieldValidationInfo"
    ) -> int | float | None:

        if interval is not None and values.data.get("values") is not None:
            raise ValueError(
                "Cannot specify both interval and values in a PropertyDomain. "
                f"Values were: {values.data.get('values')}. Interval was: {interval}"
            )

        return interval

    @pydantic.field_validator("domainRange")
    def range_requirements(
        cls,
        passed_range: list[int | float] | None,
        otherFields: "pydantic.FieldValidationInfo",
    ) -> list[int | float] | None:

        values = otherFields.data.get("values")
        if passed_range is not None and values:
            # Check if the two are compatible - this is for backwards compatibility
            result = min(passed_range) <= min(values) and max(values) < max(
                passed_range
            )
            if not result:
                raise ValueError(
                    f"Passed domainRange ({passed_range}) and values ({values} are not compatible"
                )
            # Forget the passed range
            passed_range = None

        return passed_range

    @pydantic.field_validator("variableType")
    def variableType_matches_values(
        cls, value: VariableTypeEnum, values: "pydantic.FieldValidationInfo"
    ) -> typing.Literal[
        "CONTINUOUS_VARIABLE_TYPE",
        "DISCRETE_VARIABLE_TYPE",
        "CATEGORICAL_VARIABLE_TYPE",
        "OPEN_CATEGORICAL_VARIABLE_TYPE",
        "BINARY_VARIABLE_TYPE",
        "UNKNOWN_VARIABLE_TYPE",
        "IDENTIFIER_VARIABLE_TYPE",
    ]:

        import numbers

        # If the variable type is unknown assign it
        if value == VariableTypeEnum.UNKNOWN_VARIABLE_TYPE:
            # Check if we can give a more specific type
            # Rules:
            # If values provided and all numbers == DISCRETE_VARIABLE_TYPE
            # if values provided and not all numbers == CATEGORICAL_VARIABLE_TYPE
            # if range provided and no interval == CONTINUOUS_VARIABLE_TYPE
            # if range provide  and interval == DISCRETE_VARIABLE_TYPE
            # if interval ==  DISCRETE_VARIABLE_TYPE

            if values.data.get("values") is not None:
                if all(
                    isinstance(e, numbers.Number) for e in values.data.get("values")
                ):
                    value = VariableTypeEnum.DISCRETE_VARIABLE_TYPE
                else:
                    value = VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
            elif values.data.get("domainRange") is not None:
                if values.data.get("interval") is not None:
                    value = VariableTypeEnum.DISCRETE_VARIABLE_TYPE
                else:
                    value = VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
            elif values.data.get("interval") is not None:
                value = VariableTypeEnum.DISCRETE_VARIABLE_TYPE

        if value == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:

            if values.data.get("values") is None:
                raise ValueError(
                    "The values field for a CATEGORICAL_VARIABLE_TYPE was None"
                )

            if values.data.get("interval") is not None:
                raise ValueError(
                    "The interval field for a CATEGORICAL_VARIABLE_TYPE was not None"
                )

            if (
                not all(
                    isinstance(e, numbers.Number) for e in values.data.get("values")
                )
                and values.data.get("domainRange") is not None
            ):
                raise ValueError(
                    "The domainRange field was not None for a CATEGORICAL_VARIABLE_TYPE "
                    "where the values are not all numbers"
                )

        elif value == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            # Discrete must have either values or an interval
            # If it has a range it must have an interval
            valuesCheck = values.data.get("values") is not None
            intervalCheck = values.data.get("interval") is not None
            if not (valuesCheck or intervalCheck):
                raise ValueError(
                    "A DISCRETE_VARIABLE_TYPE had neither values nor interval specified"
                )

        elif value == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:

            if values.data.get("values") is not None:
                raise ValueError(
                    "The values field of a CONTINUOUS_VARIABLE_TYPE was not None"
                )

            if values.data.get("interval") is not None:
                raise ValueError(
                    "The interval field of a CONTINUOUS_VARIABLE_TYPE was not None"
                )

        elif value == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE:

            if values.data.get("interval") is not None:
                raise ValueError(
                    "The interval field of an OPEN_CATEGORICAL_VARIABLE_TYPE was not None"
                )

            if values.data.get("domainRange") is not None:
                raise ValueError(
                    "The domainRange field of an OPEN_CATEGORICAL_VARIABLE_TYPE was not None"
                )

        return value

    @pydantic.model_serializer
    def minimize_serialization(
        self, info: "pydantic.SerializationInfo"
    ) -> dict[str, typing.Any]:

        import numbers

        from orchestrator.utilities.pydantic import (
            model_dict_representation_with_field_exclusions_for_custom_model_serializer,
        )

        dict_representation = (
            model_dict_representation_with_field_exclusions_for_custom_model_serializer(
                model=self, info=info
            )
        )

        if not info.context or not info.context.get("minimize_output", False):
            return dict_representation

        # We can remove domainRange if values are defined
        if self.values and "domainRange" in dict_representation:
            del dict_representation["domainRange"]

        # We can remove variableType according to the rules
        # defined in:
        # https://github.ibm.com/Discovery-Orchestrator/ad-orchestrator/issues/1505#issuecomment-123891159
        if "variableType" in dict_representation:
            can_delete_variable_type = False
            match self.variableType:
                case VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
                    # We can remove the variableType for categorical variables
                    # if we have values and the values are not all numbers
                    can_delete_variable_type = self.values and not all(
                        isinstance(v, numbers.Number) for v in self.values
                    )
                case VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
                    # We can remove the variableType for continuous variables
                    # if the domain range is defined
                    can_delete_variable_type = bool(self.domainRange)
                case VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
                    # We can remove the variableType for discrete variables if:
                    # - values are defined
                    # - the domain range is defined AND the interval is defined
                    #   OR
                    #   the domain range IS NOT defined AND the interval is defined
                    if self.values:
                        can_delete_variable_type = True
                    elif self.domainRange:
                        can_delete_variable_type = self.interval is not None
                    elif self.interval is not None:
                        can_delete_variable_type = True
                case VariableTypeEnum.UNKNOWN_VARIABLE_TYPE:
                    can_delete_variable_type = True
                case _:
                    # We need to always serialize:
                    # - BINARY_VARIABLE_TYPE
                    # - IDENTIFIER_VARIABLE_TYPE
                    pass

            if can_delete_variable_type:
                del dict_representation["variableType"]

        return dict_representation

    def __eq__(self, other: object) -> bool:  # noqa: ANN401
        """Two domains are considered the same if they have identical values for the properties"""

        try:
            iseq = (
                self.variableType == other.variableType
                and self.domainRange == other.domainRange
                and self.interval == other.interval
                and self.values == other.values
                and self.probabilityFunction == other.probabilityFunction
            )
        except AttributeError:
            # One of the objects is not a PropertyDomain
            iseq = False

        return iseq

    @property
    def domain_values(self) -> list:

        if self.variableType in {
            VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
            VariableTypeEnum.UNKNOWN_VARIABLE_TYPE,
            VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
        }:
            raise ValueError(
                "Cannot generate domain values for continuous, unknown or open categorical variables"
            )
        if self.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
            return [False, True]

        if self.values:
            return self.values

        return _internal_range_values(
            lower=min(self.domainRange),
            upper=max(self.domainRange),
            interval=self.interval,
        )

    def valueInDomain(self, value: typing.Any) -> bool:  # noqa: ANN401

        if self.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
            if self.domainRange is not None:
                retval = (value < max(self.domainRange)) and (
                    value >= min(self.domainRange)
                )
            else:
                import numbers

                # The domain has no range which means we just accept the value if it is a number
                retval = bool(isinstance(value, numbers.Number))
        elif self.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            if self.values:
                retval = value in self.values
            else:
                if self.domainRange is not None:
                    retval = value in self.domain_values
                else:
                    # The domain has no range or values which means we just accept the value
                    retval = True
        elif self.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
            retval = value in self.values
        elif self.variableType in [
            VariableTypeEnum.UNKNOWN_VARIABLE_TYPE,
            VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
        ]:
            # If the domain is unknown or open categorical we just return True
            # This is required if the value is from a PropertyType with these domains for self-consistency
            # e.g. If we have a ConstitutiveProperty(identifier="smiles", PropertyDomain(type=UNKNOWN_VARIABLE_TYPE)
            # And then if we ask is smiles = (CO2) in the domain it should return True.
            retval = True
        elif self.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
            retval = value in [True, False, 0, 1]
        else:  # pragma: nocover
            raise ValueError(
                f"Internal error: Unknown variable type {self.variableType}"
            )

        return retval

    def isSubDomain(self, otherDomain: "PropertyDomain") -> bool:
        """Checks if self is a subdomain of otherDomain.

        If the two domains are identical this method returns True"""

        if self is otherDomain:
            return True

        if self == otherDomain:
            return True

        # If variable types are the same, handle in each function
        if otherDomain.variableType == VariableTypeEnum.UNKNOWN_VARIABLE_TYPE:
            return is_subdomain_of_unknown_domain(
                unknownDomain=otherDomain, testDomain=self
            )
        if otherDomain.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
            return is_subdomain_of_continuous_domain(
                continuousDomain=otherDomain, testDomain=self
            )
        if otherDomain.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            return is_subdomain_of_discrete_domain(
                discreteDomain=otherDomain, testDomain=self
            )
        if otherDomain.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
            return is_subdomain_of_categorical_domain(
                categoricalDomain=otherDomain, testDomain=self
            )
        if otherDomain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
            return is_subdomain_of_binary_domain(
                binaryDomain=otherDomain, testDomain=self
            )
        if otherDomain.variableType == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE:
            return is_subdomain_of_open_categorical_domain(
                openCategoricalDomain=otherDomain, testDomain=self
            )
        # fallback to previous logic if unknown type
        raise ValueError(f"Internal error: Unknown variable type {self.variableType}")

    @property
    def size(self) -> float | int:
        """Returns the size (number of elements) in the domain if this is countable.

        Returns math.inf if the size is not countable or is unknown/open categorical.
        This includes any domain with CONTINUOUS_VARIABLE_TYPE, UNKNOWN_VARIABLE_TYPE or OPEN_CATEGORICAL_VARIABLE_TYPE.
        It also includes any unbounded domain with DISCRETE_VARIABLE_TYPE.
        """

        if (
            self.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
        ):  # noqa: SIM114
            size = math.inf
        elif (
            self.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
            and (self.domainRange is None and self.values is None)
        ) or self.variableType == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE:
            size = math.inf
        else:
            if self.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
                # If we have an interval we can use the range to get size
                # Otherwise the variable must have specified values, and we use the number of values.
                if self.interval is not None:
                    # Note: Intervals are inclusive of lower bound exclusive of upper if interval is 1
                    # If interval is greater than 1 it may include upper limit
                    # This is the same as "a_range" default behaviour and also of ray.tune.(q)randint.
                    a_range = _internal_range_values(
                        lower=min(self.domainRange),
                        upper=max(self.domainRange),
                        interval=self.interval,
                    )

                    size = len(a_range)
                else:
                    size = len(self.values)
            elif self.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
                size = 2
            elif self.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
                size = len(self.values)
            else:
                size = math.inf

        return size
