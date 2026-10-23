# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from ado.schema.property import (
    AbstractPropertyDescriptor,
    ConcretePropertyDescriptor,
    Property,
)
from ado.schema.property_value import PropertyValue
from ado.schema.reference import ExperimentReference

if typing.TYPE_CHECKING:
    from ado.schema.property import (
        MeasuredPropertyTypeEnum,
        PropertyDescriptor,
    )


class ObservedProperty(pydantic.BaseModel):
    targetProperty: Annotated[
        AbstractPropertyDescriptor | ConcretePropertyDescriptor,
        pydantic.Field(
            description="The property the receiver is an (attempted) observation of"
        ),
    ]
    experimentReference: Annotated[
        ExperimentReference,
        pydantic.Field(
            description="A reference to the experiment that produces measurements of this observed property"
        ),
    ]
    metadata: Annotated[
        dict | None,
        pydantic.Field(
            default_factory=dict,
            description="Metadata on the instance of the measurement that observed this property",
        ),
    ]
    model_config = ConfigDict(frozen=True)

    @pydantic.field_validator("targetProperty", mode="before")
    @classmethod
    def convert_property_to_descriptor(
        cls,
        value: typing.Any,  # noqa: ANN401
    ) -> "PropertyDescriptor | typing.Any":  # noqa: ANN401

        # We allow instantiation with Property models and their subclass but they are converted
        # to the equivalent descriptors
        if isinstance(value, Property):
            value = value.descriptor()

        return value

    def __eq__(self, other: object) -> bool:  # noqa: ANN401
        """Check equality with another object based on experiment reference and target property.

        Args:
            other: The object to compare with.

        Returns:
            True if other is an ObservedProperty with matching experimentReference
            and targetProperty identifier.
        """
        if not isinstance(other, ObservedProperty):
            return False
        return (
            self.experimentReference == other.experimentReference
            and self.targetProperty.identifier == other.targetProperty.identifier
        )

    def __hash__(self) -> int:
        """Compute the hash based on experiment reference and target property identifier.

        Returns:
            The integer hash value.
        """
        return hash((hash(self.experimentReference), self.targetProperty.identifier))

    @property
    def identifier(self) -> str:
        return f"{self.experimentReference.major_version_parameterized_experiment_identifier}-{self.targetProperty.identifier}"

    def __str__(self) -> str:
        return self.identifier

    def __repr__(self) -> str:
        return self.identifier

    @property
    def propertyType(self) -> "MeasuredPropertyTypeEnum":
        return self.targetProperty.propertyType


class ObservedPropertyValue(PropertyValue):
    property: Annotated[
        ObservedProperty,
        pydantic.Field(description="The ObservedProperty with the value"),
    ]
