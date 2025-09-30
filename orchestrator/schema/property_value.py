# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum

import pydantic

from orchestrator.schema.property import (
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
    Property,
    PropertyDescriptor,
)


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


class PropertyValue(pydantic.BaseModel):
    """Represents the value of a property"""

    valueType: ValueTypeEnum | None = pydantic.Field(
        default=None,
        description="The type of the value. If not set it is set based on the value.",
    )
    value: int | float | list | str | bytes | None = pydantic.Field(
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
