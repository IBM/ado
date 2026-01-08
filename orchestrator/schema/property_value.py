# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import logging
import typing
from typing import Annotated

import pydantic
from pydantic import WithJsonSchema

from orchestrator.schema.property import (
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
    Property,
    PropertyDescriptor,
)

logger = logging.getLogger("property_value")


class ValueTypeEnum(str, enum.Enum):
    NUMERIC_VALUE_TYPE = "NUMERIC_VALUE_TYPE"  # the value is a bool,int, float etc.
    VECTOR_VALUE_TYPE = "VECTOR_VALUE_TYPE"  # the value is a 1-D list or vector, possible of mixed other value types
    STRING_VALUE_TYPE = "STRING_VALUE_TYPE"  # the value is a string
    BLOB_VALUE_TYPE = "BLOB_VALUE_TYPE"  # the value is a binary blob


valueTypesDisplayNames = {
    ValueTypeEnum.NUMERIC_VALUE_TYPE: "numeric",
    ValueTypeEnum.STRING_VALUE_TYPE: "string",
    ValueTypeEnum.VECTOR_VALUE_TYPE: "vector",
    ValueTypeEnum.BLOB_VALUE_TYPE: "blob",
}


# A type to help with bytes value type + JSON + structured decoding.
# Structured decoding methods uses the model json schema to constrain value generation
# However these methods do not support fields with "binary" value types in schema
# which is what fields using "bytes" type will be annotated with
# Using this annotated type for a model field will cause its json schema not
# to use binary, but instead specify it is a base64 string
CustomBytes = Annotated[
    bytes,
    WithJsonSchema(
        # keep it as a plain string; add an optional hint for consumers
        {"type": "string", "contentEncoding": "base64"},
    ),
]


class PropertyValue(pydantic.BaseModel):
    """Represents the value of a property"""

    valueType: ValueTypeEnum | None = pydantic.Field(
        default=None,
        description="The type of the value. If not set it is set based on the value.",
    )
    value: int | float | list | str | CustomBytes | None = pydantic.Field(
        description="The measured value."
    )
    property: PropertyDescriptor | ConstitutivePropertyDescriptor = pydantic.Field(
        description="The Property with the value"
    )
    uncertainty: float | None = pydantic.Field(
        default=None, description="The uncertainty in the measured value. Can be None"
    )

    @pydantic.field_validator("property", mode="before")
    def convert_property_to_descriptor(cls, value):

        if isinstance(value, Property):
            value = value.descriptor()

        return value

    @pydantic.field_validator(
        "value",
    )
    def check_value_type(cls, value, context: pydantic.ValidationInfo):

        valueType = context.data.get("valueType")
        if valueType:
            if valueType == ValueTypeEnum.NUMERIC_VALUE_TYPE:
                if isinstance(value, str):
                    import logging

                    logger = logging.getLogger()
                    logger.warning(
                        f"TEMP: Detected string value, {value}, assigned NUMERIC_TYPE assuming due to prior bug. Will upgrade"
                    )
                elif isinstance(value, list):
                    import logging

                    logger = logging.getLogger()
                    logger.warning(
                        f"TEMP: Detected list value, {value}, assigned NUMERIC_TYPE assuming due to prior bug. Will upgrade"
                    )
                else:
                    assert type(value) in [float, int] or value is None
            elif valueType == ValueTypeEnum.STRING_VALUE_TYPE:
                assert isinstance(value, str)
            elif valueType == ValueTypeEnum.BLOB_VALUE_TYPE:
                # If type is BLOB but value is string we need to convert to bytes
                # This is because bytes are serialized in JSON as strings and if we
                # dump a byte value to JSON and then try to read it, it will fail validation unless we do this
                # Why not use pydantic.Base64Bytes as byte type as this has a build in decoder?
                # Because value is a union with string, pydantic can't tell if a string is encoded bytes or a string
                # The only impact of using Base64Bytes here would be we could use base64.b64decode
                if isinstance(value, str):
                    value = (
                        bytes(value, "utf-8").decode("unicode_escape").encode("latin1")
                    )
                else:
                    assert isinstance(value, bytes)
            elif valueType == ValueTypeEnum.VECTOR_VALUE_TYPE:
                assert isinstance(value, list)
            else:  # pragma: nocover
                raise ValueError(
                    f"No validation available for values of type {valueType}. This is an internal error. "
                )

        return value

    @pydantic.model_validator(mode="after")
    def set_value_type(self):

        if self.valueType is None:
            if type(self.value) in [float, int, type(None)]:
                self.valueType = ValueTypeEnum.NUMERIC_VALUE_TYPE
            elif isinstance(self.value, str):
                self.valueType = ValueTypeEnum.STRING_VALUE_TYPE
            elif isinstance(self.value, bytes):
                self.valueType = ValueTypeEnum.BLOB_VALUE_TYPE
            elif isinstance(self.value, list):
                self.valueType = ValueTypeEnum.VECTOR_VALUE_TYPE
        elif self.valueType == ValueTypeEnum.NUMERIC_VALUE_TYPE and isinstance(
            self.value, str
        ):
            # TEMPORARY
            self.valueType = ValueTypeEnum.STRING_VALUE_TYPE
        elif self.valueType == ValueTypeEnum.NUMERIC_VALUE_TYPE and isinstance(
            self.value, list
        ):
            # TEMPORARY
            self.valueType = ValueTypeEnum.VECTOR_VALUE_TYPE

        return self

    def __str__(self):
        return f"value-{self.property}:{self.value}"

    def __repr__(self):
        return f"value-{self.property}:{self.value}"

    def __eq__(self, other):

        return bool(
            isinstance(other, PropertyValue)
            and self.property == other.property
            and self.value == other.value
        )

    def isUncertain(self):

        return self.uncertainty is not None


class ConstitutivePropertyValue(PropertyValue):

    property: ConstitutivePropertyDescriptor = pydantic.Field(
        description="The ConstitutiveProperty with the value"
    )


def constitutive_property_values_from_point(
    point: dict, properties: list[ConstitutiveProperty | ConstitutivePropertyDescriptor]
) -> list[ConstitutivePropertyValue]:
    """Given a dict of {property id:property value}, and the Property instances, returns the PropertyValue instances"""

    return [
        ConstitutivePropertyValue(value=point[c.identifier], property=c)
        for c in properties
    ]


def validate_point_against_properties(
    point: dict[str, typing.Any],
    constitutive_properties: list[ConstitutiveProperty],
    allow_partial_matches: bool = False,
    verbose=False,
):
    """point is valid if all its keys have a constitutive_property with
    a matching identifier and all its values are in the domain of this
    property. If allow_partial_matches is False an additional condition is that
    every key in point has a matching property AND vice versa

    Params:
        point: A dictionary whose keys are property identifiers and values are values for those properties
        constitutive_properties: A list of ConstitutiveProperty instances to validate the point against
        allow_partial_matches: If True a point is valid if all its points have matching properties, even if
            there are more constitutive properties
        verbose: If True print reasons that point is not valid to stderr

    Returns:
        - True if point is compatible with space otherwise false
    """

    import sys

    constitutive_property_identifiers_for_point = set(point.keys())
    constitutive_property_identifiers_for_validation_set = {
        cp.identifier for cp in constitutive_properties
    }

    matching_constitutive_property_identifiers = (
        constitutive_property_identifiers_for_point.intersection(
            constitutive_property_identifiers_for_validation_set
        )
    )

    # If we don't allow partial matches, all properties must match
    if not allow_partial_matches and len(
        matching_constitutive_property_identifiers
    ) != len(constitutive_property_identifiers_for_validation_set):
        if verbose:
            print(
                f"The point does not contain all the constitutive properties in the validation set.\n "
                f"Missing properties: {constitutive_property_identifiers_for_validation_set - matching_constitutive_property_identifiers}.\n"
                f"Properties in validation set: {constitutive_property_identifiers_for_validation_set}.\n "
                f"Point properties matching validation set: {matching_constitutive_property_identifiers}.\n",
                file=sys.stderr,
            )
        return False

    # If we allow partial matches, all properties for the point must
    # match
    if len(matching_constitutive_property_identifiers) != len(
        constitutive_property_identifiers_for_point
    ):
        if verbose:
            print(
                f"The point contains properties not in the validation set.\n "
                f"Point properties not in validation set: {constitutive_property_identifiers_for_point - matching_constitutive_property_identifiers}.\n"
                f"Point properties in validation set: {matching_constitutive_property_identifiers}.\n ",
                file=sys.stderr,
            )
        return False

    # Once we have checked that the identifiers match, we must
    # check that the values specified by the point are in the domain
    # of the constitutive properties.
    for constitutive_property in constitutive_properties:
        if (
            constitutive_property.identifier
            not in matching_constitutive_property_identifiers
        ):
            continue

        if not constitutive_property.propertyDomain.valueInDomain(
            point[constitutive_property.identifier]
        ):
            if verbose:
                print(
                    f"Value {point[constitutive_property.identifier]} for property {constitutive_property.identifier} "
                    "is not in the domain of the matching constitutive property in the validation set "
                    f" ({constitutive_property.propertyDomain})",
                    file=sys.stderr,
                )
            return False

    return True
