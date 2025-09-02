# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pydantic
from pydantic import ConfigDict

from orchestrator.schema.property import (
    AbstractProperty,
    ConcreteProperty,
)
from orchestrator.schema.reference import ExperimentReference


class ObservedProperty(pydantic.BaseModel):
    targetProperty: AbstractProperty | ConcreteProperty = pydantic.Field(
        description="The TargetProperty the receiver is an (attempted) observation of"
    )
    experimentReference: ExperimentReference = pydantic.Field(
        description=" A reference to the experiment that produces measurements of this observed property"
    )
    metadata: dict | None = pydantic.Field(
        default={},
        description="Metadata on the instance of the measurement that observed this property",
    )
    model_config = ConfigDict(frozen=True)

    def __eq__(self, other):
        """Two properties are considered the same if they have the same identifier"""

        return self.identifier == other.identifier

    def __hash__(self):
        return hash(str(self))

    @property
    def identifier(self):
        return "%s-%s" % (
            self.experimentReference.parameterizedExperimentIdentifier,
            self.targetProperty.identifier,
        )

    def __str__(self):
        return "op-%s" % self.identifier

    def __repr__(self):
        return "op-%s" % self.identifier

    @property
    def propertyType(self):
        return self.targetProperty.propertyType
