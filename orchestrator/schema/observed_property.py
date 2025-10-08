# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pydantic
from pydantic import ConfigDict

from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConcretePropertyDescriptor,
    Property,
)
from orchestrator.schema.property_value import PropertyValue
from orchestrator.schema.reference import ExperimentReference


class ObservedProperty(pydantic.BaseModel):
    targetProperty: AbstractPropertyDescriptor | ConcretePropertyDescriptor = (
        pydantic.Field(
            description="The property the receiver is an (attempted) observation of"
        )
    )
    experimentReference: ExperimentReference = pydantic.Field(
        description=" A reference to the experiment that produces measurements of this observed property"
    )
    metadata: dict | None = pydantic.Field(
        default={},
        description="Metadata on the instance of the measurement that observed this property",
    )
    model_config = ConfigDict(frozen=True)

    @pydantic.field_validator("targetProperty", mode="before")
    @classmethod
    def convert_property_to_descriptor(cls, value):

        # We allow instantiation with Property models and their subclass but they are converted
        # to the equivalent descriptors
        if isinstance(value, Property):
            value = value.descriptor()

        return value

    def __eq__(self, other):
        """Two properties are considered the same if they have the same identifier"""

        return self.identifier == other.identifier

    def __hash__(self):
        return hash(str(self))

    @property
    def identifier(self):
        return f"{self.experimentReference.parameterizedExperimentIdentifier}-{self.targetProperty.identifier}"

    def __str__(self):
        return f"op-{self.identifier}"

    def __repr__(self):
        return f"op-{self.identifier}"

    @property
    def propertyType(self):
        return self.targetProperty.propertyType


class ObservedPropertyValue(PropertyValue):
    property: ObservedProperty = pydantic.Field(
        description="The ObservedProperty with the value"
    )
