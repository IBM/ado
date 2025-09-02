# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import typing

import pydantic
from pydantic import ConfigDict

from orchestrator.schema.domain import PropertyDomain


class MeasuredPropertyTypeEnum(str, enum.Enum):
    REPRESENTATION_PROPERTY_TYPE = "REPRESENTATION_PROPERTY_TYPE"  # These are a numerical representation of the entity they are associated with
    PHYSICAL_PROPERTY_TYPE = (
        "PHYSICAL_PROPERTY_TYPE"  # These are physical properties of a physical entity
    )
    CATEGORICAL_PROPERTY_TYPE = "CATEGORICAL_PROPERTY_TYPE"  # These are categories the entity has been placed in
    MEASURED_PROPERTY_TYPE = "MEASURED_PROPERTY_TYPE"  # A catch-all type
    OBJECTIVE_FUNCTION_PROPERTY_TYPE = "OBJECTIVE_FUNCTION_PROPERTY_TYPE"  # Properties calculated from other properties with the purpose of providing a value w.r.t to some objective


class NonMeasuredPropertyTypeEnum(str, enum.Enum):
    # Properties whose values don't require a measurement of the entity
    # Usually they are directly defined in the entities definition i.e. once have a uniquely specified the entity
    # you know these property value
    # For example if an entity is a "ResourceConfiguration" and a unique resource configuration is defined by numberCPUS
    # and numberGPUS, then numberCPUS and numberGPUS are constitutive properties

    CONSTITUTIVE_PROPERTY_TYPE = "CONSTITUTIVE_PROPERTY_TYPE"  # Properties whose values are immediately known when you define the entity


class Property(pydantic.BaseModel):
    """A named property with a domain"""

    identifier: str
    metadata: typing.Optional[dict] = pydantic.Field(
        default=None, description="Metadata on the property"
    )
    propertyDomain: PropertyDomain = pydantic.Field(
        default=PropertyDomain(),
        description="Provides information on the variable type and the valid values it can take",
    )
    model_config = ConfigDict(frozen=True, extra="forbid")

    def __eq__(self, other: "Property"):
        """Two properties are considered the same if they have the same identifier and domain.

        Metadata is not included"""

        try:
            retval = (
                self.identifier == other.identifier
                and self.propertyDomain == other.propertyDomain
            )
        except AttributeError as error:
            print(error)
            retval = False

        return retval

    def _repr_pretty_(self, p, cycle=False):

        if cycle:  # pragma: no cover
            p.text("Cycle detected")
        else:
            p.text(f"{self.identifier}")
            if self.metadata and self.metadata.get("description"):
                p.text(": " + str(self.metadata.get("description")))
            if self.propertyDomain:
                p.break_()
                with p.group(2, "Domain:"):
                    p.break_()
                    p.pretty(self.propertyDomain)

            p.breakable()


class AbstractProperty(Property):
    """Represents an Abstract Property"""

    propertyType: MeasuredPropertyTypeEnum = (
        MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )
    concretePropertyIdentifiers: typing.Optional[list[str]] = None
    model_config = ConfigDict(frozen=True)

    def __str__(self):
        return "ap-%s" % self.identifier

    def __eq__(self, other):

        retval = super().__eq__(other)
        if retval:
            retval = (
                self.concretePropertyIdentifiers == other.concretePropertyIdentifiers
            )

        return retval


class ConstitutiveProperty(Property):
    propertyType: NonMeasuredPropertyTypeEnum = pydantic.Field(
        default=NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE
    )

    def __str__(self):
        return "cp-%s" % self.identifier

    model_config = ConfigDict(frozen=True)


class ConcreteProperty(Property):
    propertyType: MeasuredPropertyTypeEnum = pydantic.Field(
        default=MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )
    abstractProperty: typing.Optional[AbstractProperty] = None
    model_config = ConfigDict(frozen=True)

    def __str__(self):
        return "cp-%s" % self.identifier
