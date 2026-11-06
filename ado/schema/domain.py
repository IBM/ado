# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
import functools
import logging
import math
import numbers
import typing
from decimal import Decimal
from typing import Annotated

import numpy as np
import pydantic
from pydantic import ConfigDict

from ado.utilities.output import format_elided_list

if typing.TYPE_CHECKING:
    from rich.console import RenderableType


class VariableTypeEnum(enum.StrEnum):
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


class ProbabilityFunctionsEnum(enum.StrEnum):
    UNIFORM = "uniform"  # A uniform distribution
    NORMAL = "normal"  # A normal distribution
    # Can easily add more


def is_float_range(
    interval: float,
    domain_range: list[int | float],
) -> bool:
    "Returns True if an on interval or domain range is a float"

    return any(isinstance(x, float) for x in [interval, *domain_range])


_DISCRETE_DOMAIN_DECIMAL_PLACES = 10
_DISCRETE_DOMAIN_SCALE_FACTOR = 10**_DISCRETE_DOMAIN_DECIMAL_PLACES


def _exceeds_supported_decimal_precision(v: float) -> bool:
    """Returns True if v has more than 10 decimal places.

    Args:
        v: The value to check.

    Returns:
        True if float(np.round(v, _DISCRETE_DOMAIN_DECIMAL_PLACES)) != v,
        False otherwise.
    """
    return float(np.round(v, _DISCRETE_DOMAIN_DECIMAL_PLACES)) != v


def _internal_range_values(lower: float, upper: float, interval: float) -> list:
    """Returns the values in the half-open [lower, upper) range.

    If all values are integers uses np.arange.
    If any value is a float computes values as lower + i * interval
    then removes the upper bound if included.

    All values are rounded to 10 decimal places to handle precision
    issues like 0.1 + 0.2 == 0.30000000000000004.

    Args:
        lower: Inclusive lower bound of the range.
        upper: Exclusive upper bound of the range.
        interval: Step size between consecutive values.

    Returns:
        List of evenly-spaced values in [lower, upper), rounded to 10
        decimal places.

    Raises:
        ValueError: If interval is not greater than 0.
    """
    if interval <= 0:
        raise ValueError(f"interval must be greater than 0. Got: {interval}")

    if not is_float_range(interval=interval, domain_range=[lower, upper]):
        return np.arange(lower, upper, interval).tolist()

    step_count = int(np.floor((upper - lower) / interval)) + 1
    values = np.round(
        [lower + step_index * interval for step_index in range(step_count)],
        _DISCRETE_DOMAIN_DECIMAL_PLACES,
    ).tolist()
    while values and values[-1] >= upper:
        values.pop()

    return values


def _to_scaled_int(
    value: typing.Any,  # noqa: ANN401
) -> int:
    """Multiplies value by a scale factor that turns it into an exact integer.

    The scale factor is 10**_DISCRETE_DOMAIN_DECIMAL_PLACES (currently 10**10).

    Args:
        value: An int, float, bool or NumPy scalar with at most
            _DISCRETE_DOMAIN_DECIMAL_PLACES decimal places.

    Returns:
        value * _DISCRETE_DOMAIN_SCALE_FACTOR.

    Raises:
        ValueError: If value is not an int or float, is not finite, or has more
            than _DISCRETE_DOMAIN_DECIMAL_PLACES decimal places.
    """
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        value = int(value)

    if isinstance(value, int):
        return value * _DISCRETE_DOMAIN_SCALE_FACTOR

    if not isinstance(value, float):
        raise ValueError(f"value must be an int or float. Got: {value!r}")

    if not math.isfinite(value):
        raise ValueError(f"value must be finite. Got: {value!r}")

    scaled_value = Decimal(repr(value)).scaleb(_DISCRETE_DOMAIN_DECIMAL_PLACES)
    if scaled_value != scaled_value.to_integral_value():
        raise ValueError(
            f"value {value!r} exceeds the maximum supported precision of "
            f"{_DISCRETE_DOMAIN_DECIMAL_PLACES} decimal places"
        )

    return int(scaled_value)


_UNSET: typing.Any = object()


class ScaledDiscreteDomainRange(pydantic.BaseModel):
    """Represents a discrete domain range scaled to exact integer coordinates.

    Coordinates are scaled by 10**10 to preserve decimal precision without
    floating-point arithmetic drift.
    """

    lower: Annotated[
        int,
        pydantic.Field(
            description="Inclusive lower bound in scaled integer coordinates"
        ),
    ]
    upper: Annotated[
        int,
        pydantic.Field(
            description="Exclusive upper bound in scaled integer coordinates"
        ),
    ]
    interval: Annotated[
        int,
        pydantic.Field(
            gt=0,
            description="Step size between values in scaled integer coordinates",
        ),
    ]

    model_config = ConfigDict(frozen=True, extra="forbid")

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def size(self) -> int:
        """Returns the number of elements in the range."""
        span = self.upper - self.lower
        if span <= 0:
            return 0
        # Exact integer ceiling division ceil(span / interval) without floating-point drift
        return (span + self.interval - 1) // self.interval

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @functools.cached_property
    def last(self) -> int | None:
        """Returns the last element in the range, or None if the range is empty."""
        if self.size == 0:
            return None
        return self.lower + (self.size - 1) * self.interval

    @classmethod
    def from_unscaled(
        cls, lower: float, upper: float, interval: float
    ) -> "ScaledDiscreteDomainRange":
        """Creates a ScaledDiscreteDomainRange from unscaled range bounds.

        Args:
            lower: Inclusive lower bound of the range.
            upper: Exclusive upper bound of the range.
            interval: Step size between consecutive values.

        Returns:
            A ScaledDiscreteDomainRange with exact integer coordinates.

        Raises:
            ValueError: If interval is not greater than 0, or a value has more
                than 10 decimal places.
        """
        if interval <= 0:
            raise ValueError(f"interval must be greater than 0. Got: {interval}")

        return cls(
            lower=_to_scaled_int(lower),
            upper=_to_scaled_int(upper),
            interval=_to_scaled_int(interval),
        )

    def contains(
        self,
        *,
        value: typing.Any = _UNSET,  # noqa: ANN401
        scaled_value: int | None = None,
    ) -> bool:
        """Returns True if the given value is an element of this scaled range.

        Exactly one of value and scaled_value must be provided.

        Args:
            value: Unscaled candidate value to test.
            scaled_value: Candidate value already in scaled integer coordinates.

        Returns:
            True if the value lies within the bounds and is on the grid. False
            if value is not a finite number with at most 10 decimal places.

        Raises:
            ValueError: If both or neither of value and scaled_value are provided.
        """
        if (value is _UNSET) == (scaled_value is None):
            raise ValueError("Exactly one of value and scaled_value must be provided")

        if scaled_value is None:
            try:
                scaled_value = _to_scaled_int(value)
            except ValueError:
                return False

        return (
            self.lower <= scaled_value < self.upper
            and (scaled_value - self.lower) % self.interval == 0
        )

    def overlaps(self, other: "ScaledDiscreteDomainRange") -> bool:
        """Returns True if this range and the other share at least one element.

        Finds the smallest value that is an element of the infinite progressions
        of both ranges and above both lower bounds, by solving the congruence
        self.lower + a * self.interval == other.lower + b * other.interval.
        The ranges overlap if that value is not above both last elements.

        Args:
            other: The range to compare against.

        Returns:
            True if the ranges share an element.
        """
        # An empty range has no elements to share.
        if self.last is None or other.last is None:
            return False

        # Any shared element lies in [shared_lower_limit, shared_upper_limit],
        # the window covered by both ranges.
        shared_lower_limit = max(self.lower, other.lower)
        shared_upper_limit = min(self.last, other.last)
        if shared_lower_limit > shared_upper_limit:
            return False

        # self.lower + a * self.interval == other.lower + b * other.interval
        # has an integer solution only if the gcd of the intervals divides the
        # difference of the lower bounds.
        intervals_gcd = math.gcd(self.interval, other.interval)
        lower_bounds_difference = other.lower - self.lower
        if lower_bounds_difference % intervals_gcd != 0:
            return False

        # Chinese Remainder Theorem: solves the congruence for the smallest
        # non-negative step count of this range that lands on the other
        # range's progression.
        reduced_self_interval = self.interval // intervals_gcd
        reduced_other_interval = other.interval // intervals_gcd
        if reduced_other_interval == 1:
            self_step_count = 0
        else:
            self_step_count = (
                lower_bounds_difference
                // intervals_gcd
                * pow(reduced_self_interval, -1, reduced_other_interval)
            ) % reduced_other_interval

        # The smallest value at or above self.lower on both infinite
        # progressions. It can lie below other.lower.
        smallest_shared_value = self.lower + self_step_count * self.interval

        # Shared values repeat every lcm of the intervals. If the smallest shared
        # value is below the window, advance it by the fewest whole periods that
        # reach the window.
        shared_values_period = reduced_self_interval * other.interval
        if smallest_shared_value < shared_lower_limit:
            distance_below_limit = shared_lower_limit - smallest_shared_value
            steps_to_reach_lower_limit = (
                distance_below_limit + shared_values_period - 1
            ) // shared_values_period
            smallest_shared_value += steps_to_reach_lower_limit * shared_values_period

        # The ranges overlap if the smallest shared value inside the window does
        # not exceed the window's upper limit.
        return smallest_shared_value <= shared_upper_limit

    def is_subdomain_of(self, other: "ScaledDiscreteDomainRange") -> bool:
        """Returns True if every element of this range is an element of the other.

        This range is a subset of the other if it is empty, or if its first and
        last elements are elements of the other and, when it has more than one
        element, its interval is a multiple of the other's interval.

        Args:
            other: The enclosing range.

        Returns:
            True if this range is a subset of the other.
        """
        if self.size == 0:
            return True

        if not (
            other.contains(scaled_value=self.lower)
            and other.contains(scaled_value=self.last)
        ):
            return False

        return self.size == 1 or self.interval % other.interval == 0


def _is_multiple_of_discrete_domain_interval(
    value: typing.Any,  # noqa: ANN401
    interval: float,
) -> bool:
    """Returns True if value is an exact multiple of interval.

    Values and intervals are scaled to integers at 10 decimal places so that
    the modulo is exact for decimal values.

    Args:
        value: Candidate value to test. Must be a number or numpy scalar.
        interval: Step size of the grid.

    Returns:
        True if value is finite and an exact multiple of interval, False
        otherwise.

    Raises:
        ValueError: If interval is not greater than 0, or if value is not a
            number or numpy scalar.
    """
    if interval <= 0:
        raise ValueError(f"interval must be greater than 0. Got: {interval}")

    if not isinstance(value, (numbers.Number, np.generic)):
        raise ValueError(f"value must be a number. Got: {value!r}")

    if np.isnan(value) or np.isinf(value):
        return False

    try:
        scaled_value = _to_scaled_int(value)
    except ValueError:
        return False

    return scaled_value % _to_scaled_int(interval) == 0


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


def _is_subdomain_of_unbounded_discrete_domain(
    discreteDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if testDomain is a subdomain of an unbounded discrete domain.

    An unbounded discrete domain has interval=k and no domainRange or values.
    testDomain is a subdomain if and only if every value it can take is a
    multiple of k. Non-numeric values are never multiples of k.

    Args:
        discreteDomain: An unbounded PropertyDomain with variableType
            DISCRETE_VARIABLE_TYPE.
        testDomain: A PropertyDomain with any variableType.

    Returns:
        True if testDomain is a subdomain of discreteDomain, False otherwise.
    """
    k = discreteDomain.interval
    if k is None:
        return False

    if testDomain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
        return all(_is_multiple_of_discrete_domain_interval(x, k) for x in (0, 1))

    if testDomain.variableType != VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
        return False

    if testDomain.values is not None:
        # valueInDomain returns False for non-numeric values
        return all(discreteDomain.valueInDomain(v) for v in testDomain.values)

    if testDomain.size == 1:
        return _is_multiple_of_discrete_domain_interval(testDomain.domain_values[0], k)

    if testDomain.interval is None:
        return False

    if testDomain.size == math.inf:
        return _is_multiple_of_discrete_domain_interval(testDomain.interval, k)

    # A range with several values: the first value and the step are both
    # multiples of k.
    if testDomain.domainRange is None:
        return False
    return _is_multiple_of_discrete_domain_interval(
        min(testDomain.domainRange), k
    ) and _is_multiple_of_discrete_domain_interval(testDomain.interval, k)


def _is_subdomain_of_bounded_discrete_domain(
    discreteDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if testDomain is a subdomain of a bounded discrete domain.

    A bounded discrete domain has a finite set of values, given either
    explicitly or by a domainRange and interval. testDomain is a subdomain if
    the values it can take are contained in that set.

    Args:
        discreteDomain: A bounded PropertyDomain with variableType
            DISCRETE_VARIABLE_TYPE.
        testDomain: A PropertyDomain with any variableType.

    Returns:
        True if testDomain is a subdomain of discreteDomain, False otherwise.
    """
    if testDomain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
        return all(discreteDomain.valueInDomain(x) for x in (0, 1))

    if testDomain.variableType != VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
        return False

    # An unbounded domain is not contained in a bounded one
    if testDomain.size == math.inf:
        return False

    discrete_domain_is_defined_by_range = discreteDomain.interval is not None
    test_domain_is_defined_by_range = testDomain.interval is not None

    if discrete_domain_is_defined_by_range and test_domain_is_defined_by_range:
        return testDomain.scaled_discrete_domain_range().is_subdomain_of(
            discreteDomain.scaled_discrete_domain_range()
        )

    if discrete_domain_is_defined_by_range:
        return all(discreteDomain.valueInDomain(v) for v in testDomain.values)

    if test_domain_is_defined_by_range:
        # More elements than the enclosing domain has values cannot be a subset
        if testDomain.size > len(discreteDomain.values):
            return False
        return set(testDomain.domain_values).issubset(discreteDomain.values)

    return set(testDomain.values).issubset(discreteDomain.values)


def is_subdomain_of_discrete_domain(
    discreteDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if testDomain is a subdomain of discreteDomain.

    Args:
        discreteDomain: A PropertyDomain with variableType DISCRETE_VARIABLE_TYPE.
        testDomain: A PropertyDomain with any variableType.

    Returns:
        True if testDomain is a subdomain of discreteDomain, False otherwise.
    """
    if discreteDomain.size == math.inf:
        return _is_subdomain_of_unbounded_discrete_domain(discreteDomain, testDomain)
    return _is_subdomain_of_bounded_discrete_domain(discreteDomain, testDomain)


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


def overlaps_with_continuous_domain(
    continuousDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if testDomain shares at least one point with continuousDomain.

    Parameters:
        continuousDomain: A PropertyDomain with variableType CONTINUOUS_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the domains overlap
    """
    match testDomain.variableType:
        case (
            VariableTypeEnum.UNKNOWN_VARIABLE_TYPE
            | VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
        ):
            return True
        case VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
            return False
        case VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
            if continuousDomain.domainRange is None or testDomain.domainRange is None:
                return True
            # [a, b) ∩ [c, d) != empty  iff  a < d and c < b
            return min(continuousDomain.domainRange) < max(
                testDomain.domainRange
            ) and min(testDomain.domainRange) < max(continuousDomain.domainRange)
        case VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            if continuousDomain.domainRange is None:
                return True
            if testDomain.size == math.inf:
                # Unbounded discrete overlaps any continuous
                return True
            # Check if at least one discrete value falls in [a, b)
            return any(
                min(continuousDomain.domainRange)
                <= v
                < max(continuousDomain.domainRange)
                for v in testDomain.domain_values
            )
        case _:
            # BINARY: check if 0 or 1 is within the continuous range
            if continuousDomain.domainRange is None:
                return True
            return any(
                min(continuousDomain.domainRange)
                <= v
                < max(continuousDomain.domainRange)
                for v in [0, 1]
            )


def overlaps_with_discrete_domain(
    discreteDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if testDomain shares at least one point with discreteDomain.

    Parameters:
        discreteDomain: A PropertyDomain with variableType DISCRETE_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the domains overlap
    """
    match testDomain.variableType:
        case (
            VariableTypeEnum.UNKNOWN_VARIABLE_TYPE
            | VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
        ):
            return True
        case VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
            return overlaps_with_continuous_domain(
                continuousDomain=testDomain, testDomain=discreteDomain
            )
        case VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            if discreteDomain.size == math.inf or testDomain.size == math.inf:
                # Unbounded discrete domains overlap
                return True
            discrete_domain_is_defined_by_range = discreteDomain.interval is not None
            test_domain_is_defined_by_range = testDomain.interval is not None
            if discrete_domain_is_defined_by_range and test_domain_is_defined_by_range:
                return discreteDomain.scaled_discrete_domain_range().overlaps(
                    testDomain.scaled_discrete_domain_range()
                )
            if discrete_domain_is_defined_by_range:
                return any(discreteDomain.valueInDomain(v) for v in testDomain.values)
            if test_domain_is_defined_by_range:
                return any(testDomain.valueInDomain(v) for v in discreteDomain.values)
            return set(discreteDomain.values).isdisjoint(testDomain.values) is False
        case VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
            if discreteDomain.size == math.inf:
                return True
            return (
                set(discreteDomain.domain_values).isdisjoint(testDomain.domain_values)
                is False
            )
        case _:
            # BINARY: check if 0 or 1 is in the discrete domain
            if discreteDomain.size == math.inf:
                return True
            return any(v in discreteDomain.domain_values for v in [0, 1])


def overlaps_with_categorical_domain(
    categoricalDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if testDomain shares at least one point with categoricalDomain.

    Parameters:
        categoricalDomain: A PropertyDomain with variableType CATEGORICAL_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the domains overlap
    """
    match testDomain.variableType:
        case (
            VariableTypeEnum.UNKNOWN_VARIABLE_TYPE
            | VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
        ):
            return True
        case VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
            return False
        case VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
            return set(categoricalDomain.values).isdisjoint(testDomain.values) is False
        case VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            if testDomain.size == math.inf:
                return True
            return (
                set(categoricalDomain.domain_values).isdisjoint(
                    testDomain.domain_values
                )
                is False
            )
        case _:
            # BINARY: check if 0 or 1 appears in the categorical values
            return any(v in categoricalDomain.domain_values for v in [0, 1])


def overlaps_with_binary_domain(
    binaryDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True if testDomain shares at least one point with binaryDomain.

    Parameters:
        binaryDomain: A PropertyDomain with variableType BINARY_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True if the domains overlap
    """
    match testDomain.variableType:
        case (
            VariableTypeEnum.UNKNOWN_VARIABLE_TYPE
            | VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
            | VariableTypeEnum.BINARY_VARIABLE_TYPE
        ):
            return True
        case VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
            return overlaps_with_continuous_domain(
                continuousDomain=testDomain, testDomain=binaryDomain
            )
        case VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            if testDomain.size == math.inf:
                return True
            return any(v in testDomain.domain_values for v in [0, 1])
        case _:
            # CATEGORICAL
            return any(v in testDomain.domain_values for v in [0, 1])


def overlaps_with_unknown_domain(
    unknownDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True always — UNKNOWN domains overlap with any domain.

    Parameters:
        unknownDomain: A PropertyDomain with variableType UNKNOWN_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True
    """
    return True


def overlaps_with_open_categorical_domain(
    openCategoricalDomain: "PropertyDomain", testDomain: "PropertyDomain"
) -> bool:
    """Returns True always — OPEN_CATEGORICAL domains overlap with any domain.

    Parameters:
        openCategoricalDomain: A PropertyDomain with variableType OPEN_CATEGORICAL_VARIABLE_TYPE
        testDomain: A PropertyDomain with any variableType
    Returns:
        True
    """
    return True


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
            gt=0,
            description="The interval between discrete values variables. Do not set if values is set",
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

        from ado.utilities.rich import get_rich_repr

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
        """Validates the interval of the domain.

        Args:
            interval: The interval between discrete values, or None.
            values: Validation info holding the fields validated so far.

        Returns:
            The unchanged interval.

        Raises:
            ValueError: If values is also set, or if interval is a float with
                more than 10 decimal places.
        """

        if interval is not None and values.data.get("values") is not None:
            raise ValueError(
                "Cannot specify both interval and values in a PropertyDomain. "
                f"Values were: {values.data.get('values')}. Interval was: {interval}"
            )

        if isinstance(interval, float) and _exceeds_supported_decimal_precision(
            interval
        ):
            raise ValueError(
                f"interval value {interval} exceeds the maximum supported "
                "precision of 10 decimal places. Provide a value with at "
                "most 10 decimal places."
            )

        return interval

    @pydantic.field_validator("domainRange")
    def range_requirements(
        cls,
        passed_range: list[int | float] | None,
        otherFields: "pydantic.FieldValidationInfo",
    ) -> list[int | float] | None:
        """Validates the range of the domain.

        Args:
            passed_range: The lower (inclusive) and upper (exclusive) bounds, or None.
            otherFields: Validation info holding the fields validated so far.

        Returns:
            The passed range, or None if values is set and the range is compatible
            with it.

        Raises:
            ValueError: If the range is incompatible with values, or if a float
                bound has more than 10 decimal places.
        """

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

        for bound in passed_range or []:
            if isinstance(bound, float) and _exceeds_supported_decimal_precision(bound):
                raise ValueError(
                    f"domainRange value {bound} exceeds the maximum supported "
                    "precision of 10 decimal places. Provide a value with at "
                    "most 10 decimal places."
                )

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
                    isinstance(e, (numbers.Number, np.generic))
                    for e in values.data.get("values")
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

        elif value == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            # Discrete must have either values or an interval
            # If it has a range it must have an interval
            valuesCheck = values.data.get("values") is not None
            intervalCheck = values.data.get("interval") is not None
            if not (valuesCheck or intervalCheck):
                domain_range = values.data.get("domainRange")
                raise ValueError(
                    f"A DISCRETE_VARIABLE_TYPE had neither values nor interval specified. "
                    f"Provided: values={values.data.get('values')}, "
                    f"interval={values.data.get('interval')}, "
                    f"domainRange={domain_range}"
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
                    f"An OPEN_CATEGORICAL_VARIABLE_TYPE must not have interval specified. "
                    f"Provided: interval={values.data.get('interval')}"
                )

            if values.data.get("domainRange") is not None:
                raise ValueError(
                    f"An OPEN_CATEGORICAL_VARIABLE_TYPE must not have domainRange specified. "
                    f"Provided: domainRange={values.data.get('domainRange')}"
                )

        elif value == VariableTypeEnum.BINARY_VARIABLE_TYPE:
            # Binary variables should not have values, interval, or domainRange specified
            # They always default to [False, True].
            # To avoid breaking too many things, however, we accept [False, True]
            binary_values = values.data.get("values")
            if binary_values is not None and sorted(binary_values) != [False, True]:
                raise ValueError(
                    "BINARY_VARIABLE_TYPE should not have values specified. "
                    "It always defaults to [False, True]. "
                    "If you want to restrict the domain to specific boolean values, "
                    "use DISCRETE_VARIABLE_TYPE with values [0] or [1] instead."
                )

            if values.data.get("interval") is not None:
                raise ValueError(
                    f"A BINARY_VARIABLE_TYPE must not have interval specified. "
                    f"Provided: interval={values.data.get('interval')}"
                )

            if values.data.get("domainRange") is not None:
                raise ValueError(
                    f"A BINARY_VARIABLE_TYPE must not have domainRange specified. "
                    f"Provided: domainRange={values.data.get('domainRange')}"
                )

        return value

    @pydantic.model_serializer
    def minimize_serialization(
        self, info: "pydantic.SerializationInfo"
    ) -> dict[str, typing.Any]:

        from ado.utilities.pydantic import (
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
                        isinstance(v, (numbers.Number, np.generic)) for v in self.values
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

        if self.domainRange is None:
            raise ValueError(
                "Cannot generate domain values for an unbounded discrete domain"
            )

        return _internal_range_values(
            lower=min(self.domainRange),
            upper=max(self.domainRange),
            interval=self.interval,
        )

    def scaled_discrete_domain_range(self) -> ScaledDiscreteDomainRange:
        """Returns the scaled integer representation of this domain's range.

        Returns:
            The ScaledDiscreteDomainRange of this domain's domainRange and interval.

        Raises:
            ValueError: If the domain is not a discrete domain defined by a
                domainRange and an interval.
        """
        if (
            self.variableType != VariableTypeEnum.DISCRETE_VARIABLE_TYPE
            or self.domainRange is None
            or self.interval is None
        ):
            raise ValueError(
                "Only a discrete domain defined by a domainRange and an interval "
                f"has a scaled discrete domain range. Got: {self!r}"
            )

        return ScaledDiscreteDomainRange.from_unscaled(
            lower=min(self.domainRange),
            upper=max(self.domainRange),
            interval=self.interval,
        )

    def valueInDomain(self, value: typing.Any) -> bool:  # noqa: ANN401

        if self.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
            if self.domainRange is not None:
                retval = bool(
                    (value < max(self.domainRange)) and (value >= min(self.domainRange))
                )
            else:
                # The domain has no range which means we just accept the value if it is a number or numpy scalar
                retval = bool(isinstance(value, (numbers.Number, np.generic)))
        elif self.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            if self.values:
                retval = value in self.values
            else:
                if self.domainRange is not None:
                    retval = self.scaled_discrete_domain_range().contains(value=value)
                else:
                    retval = isinstance(
                        value, (numbers.Number, np.generic)
                    ) and _is_multiple_of_discrete_domain_interval(
                        value=value,
                        interval=self.interval,
                    )
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

    def overlaps(self, other: "PropertyDomain") -> bool:
        """Returns True if self and other share at least one point.

        Parameters:
            other: A PropertyDomain to compare against
        Returns:
            True if the domains overlap (i.e. share at least one value)
        """
        if other.variableType == VariableTypeEnum.UNKNOWN_VARIABLE_TYPE:
            return overlaps_with_unknown_domain(unknownDomain=other, testDomain=self)
        if other.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
            return overlaps_with_continuous_domain(
                continuousDomain=other, testDomain=self
            )
        if other.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
            return overlaps_with_discrete_domain(discreteDomain=other, testDomain=self)
        if other.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
            return overlaps_with_categorical_domain(
                categoricalDomain=other, testDomain=self
            )
        if other.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
            return overlaps_with_binary_domain(binaryDomain=other, testDomain=self)
        if other.variableType == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE:
            return overlaps_with_open_categorical_domain(
                openCategoricalDomain=other, testDomain=self
            )
        raise ValueError(f"Internal error: Unknown variable type {other.variableType}")

    @property
    def size(self) -> float | int:
        """Returns the size (number of elements) in the domain if this is countable.

        Returns math.inf if the size is not countable or is unknown/open categorical.
        This includes any domain with CONTINUOUS_VARIABLE_TYPE, UNKNOWN_VARIABLE_TYPE or OPEN_CATEGORICAL_VARIABLE_TYPE.
        It also includes any unbounded domain with DISCRETE_VARIABLE_TYPE.
        """

        if self.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:  # noqa: SIM114
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
                    size = self.scaled_discrete_domain_range().size
                else:
                    size = len(self.values)
            elif self.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE:
                size = 2
            elif self.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
                size = len(self.values)
            else:
                size = math.inf

        return size

    def compact_representation(self, max_items: int | None = None) -> str:
        """Return a compact string representation of this domain.

        Args:
            max_items: Maximum number of items to show before truncating with a
                ``(+N more)`` suffix.  Pass ``None`` to disable truncation.

        Returns:
            A compact string such as ``[270, 300)``, ``[0, 20) @ 5``,
            ``[0, 1, 5]``, or ``['A', 'B', ..., 'Y', 'Z'] (+10 more)``.
        """

        match self.variableType:
            case VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
                if self.domainRange is not None:
                    range_start, range_end = (
                        min(self.domainRange),
                        max(self.domainRange),
                    )
                    return f"[{range_start}, {range_end})"
                return "-"

            case VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
                if self.interval is not None and self.domainRange is not None:
                    range_start, range_end = (
                        min(self.domainRange),
                        max(self.domainRange),
                    )
                    return f"[{range_start}, {range_end}) @ {self.interval}"
                if self.values is not None:
                    return format_elided_list(self.values, max_items)
                return "-"

            case (
                VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
                | VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
            ):
                items = self.values if self.values is not None else []
                return format_elided_list(items, max_items)

            case VariableTypeEnum.BINARY_VARIABLE_TYPE:
                return "['False', 'True']"

            case VariableTypeEnum.IDENTIFIER_VARIABLE_TYPE:
                if self.values is not None:
                    return format_elided_list(self.values, max_items)
                return "-"

            case _:
                # UNKNOWN_VARIABLE_TYPE or any future type
                return "-"
