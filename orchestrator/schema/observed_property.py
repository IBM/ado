# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConcretePropertyDescriptor,
    Property,
)
from orchestrator.schema.property_value import PropertyValue
from orchestrator.schema.reference import ExperimentReference

if typing.TYPE_CHECKING:
    from orchestrator.schema.property import (
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
        cls, value: typing.Any  # noqa: ANN401
    ) -> "PropertyDescriptor | typing.Any":  # noqa: ANN401

        # We allow instantiation with Property models and their subclass but they are converted
        # to the equivalent descriptors
        if isinstance(value, Property):
            value = value.descriptor()

        return value

    def __eq__(self, other: object) -> bool:  # noqa: ANN401
        """Two properties are considered the same if they have the same identifier"""

        return self.identifier == other.identifier

    def __hash__(self) -> int:
        return hash(str(self))

    @property
    def identifier(self) -> str:
        return f"{self.experimentReference.parameterizedExperimentIdentifier}-{self.targetProperty.identifier}"

    def __str__(self) -> str:
        return f"op-{self.identifier}"

    def __repr__(self) -> str:
        return f"op-{self.identifier}"

    @property
    def propertyType(self) -> "MeasuredPropertyTypeEnum":
        return self.targetProperty.propertyType


class ObservedPropertyValue(PropertyValue):
    property: Annotated[
        ObservedProperty,
        pydantic.Field(description="The ObservedProperty with the value"),
    ]
