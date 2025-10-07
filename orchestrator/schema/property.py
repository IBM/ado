# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum

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


class PropertyDescriptor(pydantic.BaseModel):
    """A named property - no domain"""

    identifier: str
    model_config = ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="before")
    @classmethod
    def property_to_descriptor(cls, value):

        if isinstance(value, Property):
            value = value.descriptor()
        elif isinstance(value, dict):
            value.pop("propertyDomain", None)
            value.pop("metadata", None)

        return value

    def __eq__(self, other: "Property"):
        """Two PropertyDescriptors are considered the same if they have the same identifier

        A PropertyDescriptor will be equal to a Property if it has the same identifier.

        Metadata is not included"""
        return hasattr(other, "identifier") and self.identifier == other.identifier

    def _repr_pretty_(self, p, cycle=False):

        if cycle:  # pragma: no cover
            p.text("Cycle detected")
        else:
            p.text(f"{self.identifier}")
            p.breakable()


class AbstractPropertyDescriptor(PropertyDescriptor):

    propertyType: MeasuredPropertyTypeEnum = (
        MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def property_to_descriptor(cls, value):

        if isinstance(value, Property):
            value = value.descriptor()
        elif isinstance(value, dict):
            value.pop("propertyDomain", None)
            value.pop("metadata", None)
            value.pop("concretePropertyIdentifiers", None)

        return value

    def __str__(self):
        return f"ap-{self.identifier}"


class ConstitutivePropertyDescriptor(PropertyDescriptor):
    propertyType: NonMeasuredPropertyTypeEnum = pydantic.Field(
        default=NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE
    )

    def __str__(self):
        return f"cp-{self.identifier}"

    model_config = ConfigDict(frozen=True)


class ConcretePropertyDescriptor(PropertyDescriptor):

    propertyType: MeasuredPropertyTypeEnum = pydantic.Field(
        default=MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )
    abstractProperty: AbstractPropertyDescriptor | None = None
    model_config = ConfigDict(frozen=True)

    def __str__(self):
        return f"cp-{self.identifier}"


class Property(pydantic.BaseModel):
    """A named property with a domain"""

    identifier: str
    metadata: dict | None = pydantic.Field(
        default=None, description="Metadata on the property"
    )
    propertyDomain: PropertyDomain = pydantic.Field(
        default=PropertyDomain(),
        description="Provides information on the variable type and the valid values it can take",
    )
    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_descriptor(cls, descriptor: PropertyDescriptor):

        return cls(identifier=descriptor.identifier)

    def __eq__(self, other: "Property"):
        """Two properties are considered the same if they have the same identifier and domain.

        Metadata is not included"""

        try:
            retval = (
                self.identifier == other.identifier
                and self.propertyDomain == other.propertyDomain
            )
        except AttributeError:
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

    def descriptor(self):

        return PropertyDescriptor(identifier=self.identifier)


class AbstractProperty(Property):
    """Represents an Abstract Property"""

    propertyType: MeasuredPropertyTypeEnum = (
        MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )
    concretePropertyIdentifiers: list[str] | None = None
    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_descriptor(cls, descriptor: AbstractPropertyDescriptor):

        return cls(
            identifier=descriptor.identifier,
        )

    def __str__(self):
        return f"ap-{self.identifier}"

    def __eq__(self, other):

        retval = super().__eq__(other)
        return (
            retval
            and hasattr(other, "concretePropertyIdentifiers")
            and self.concretePropertyIdentifiers == other.concretePropertyIdentifiers
        )

    def descriptor(self):

        return AbstractPropertyDescriptor(identifier=self.identifier)


class ConstitutiveProperty(Property):
    propertyType: NonMeasuredPropertyTypeEnum = pydantic.Field(
        default=NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE
    )

    @classmethod
    def from_descriptor(cls, descriptor: AbstractPropertyDescriptor):

        return cls(
            identifier=descriptor.identifier,
        )

    def __str__(self):
        return f"cp-{self.identifier}"

    model_config = ConfigDict(frozen=True)

    def descriptor(self):

        return ConstitutivePropertyDescriptor(identifier=self.identifier)


class ConcreteProperty(Property):
    propertyType: MeasuredPropertyTypeEnum = pydantic.Field(
        default=MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )
    abstractProperty: AbstractProperty | None = None
    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_descriptor(cls, descriptor: ConcretePropertyDescriptor):

        return cls(
            identifier=descriptor.identifier,
            abstractProperty=(
                AbstractProperty.from_descriptor(descriptor.abstractProperty)
                if descriptor.abstractProperty
                else None
            ),
        )

    def __str__(self):
        return f"cp-{self.identifier}"

    def descriptor(self):
        return ConcretePropertyDescriptor(
            identifier=self.identifier,
            abstractProperty=(
                self.abstractProperty.descriptor() if self.abstractProperty else None
            ),
        )
