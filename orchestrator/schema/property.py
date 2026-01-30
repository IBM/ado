# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.schema.domain import PropertyDomain

if typing.TYPE_CHECKING:
    from rich.console import RenderableType


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
    def property_to_descriptor(
        cls, value: typing.Any  # noqa: ANN401
    ) -> "PropertyDescriptor | dict | typing.Any":  # noqa: ANN401

        if isinstance(value, Property):
            value = value.descriptor()
        elif isinstance(value, dict):
            value.pop("propertyDomain", None)
            value.pop("metadata", None)

        return value

    def __eq__(self, other: object) -> bool:
        """Two PropertyDescriptors are considered the same if they have the same identifier

        A PropertyDescriptor will be equal to a Property if it has the same identifier.

        Metadata is not included"""

        return (
            isinstance(other, (Property, PropertyDescriptor))
            and self.identifier == other.identifier
        )

    def __rich__(self) -> "RenderableType":
        """Render this property descriptor using rich."""
        from rich.text import Text

        return Text(self.identifier)


class AbstractPropertyDescriptor(PropertyDescriptor):

    propertyType: MeasuredPropertyTypeEnum = (
        MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def property_to_descriptor(
        cls, value: typing.Any  # noqa: ANN401
    ) -> PropertyDescriptor | dict | typing.Any:  # noqa: ANN401

        if isinstance(value, Property):
            value = value.descriptor()
        elif isinstance(value, dict):
            value.pop("propertyDomain", None)
            value.pop("metadata", None)
            value.pop("concretePropertyIdentifiers", None)

        return value

    def __str__(self) -> str:
        return f"ap-{self.identifier}"


class ConstitutivePropertyDescriptor(PropertyDescriptor):
    propertyType: Annotated[NonMeasuredPropertyTypeEnum, pydantic.Field()] = (
        NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE
    )

    def __str__(self) -> str:
        return f"cp-{self.identifier}"

    model_config = ConfigDict(frozen=True)


class ConcretePropertyDescriptor(PropertyDescriptor):

    propertyType: Annotated[MeasuredPropertyTypeEnum, pydantic.Field()] = (
        MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )
    abstractProperty: AbstractPropertyDescriptor | None = None
    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        return f"cp-{self.identifier}"


class Property(pydantic.BaseModel):
    """A named property with a domain"""

    identifier: Annotated[str, pydantic.Field()]
    metadata: Annotated[
        dict | None, pydantic.Field(description="Metadata on the property")
    ] = None
    propertyDomain: Annotated[
        PropertyDomain,
        pydantic.Field(
            description="Provides information on the variable type and the valid values it can take"
        ),
    ] = PropertyDomain()
    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_descriptor(cls, descriptor: PropertyDescriptor) -> "Property":

        return cls(identifier=descriptor.identifier)

    def __eq__(self, other: object) -> bool:  # noqa: ANN401
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

    def __rich__(self) -> "RenderableType":
        """Render this property using rich."""
        import rich.box
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        content = [
            Text.assemble(
                ("Identifier: ", "bold"),
                (self.identifier, "bold green"),
                overflow="fold",
            ),
        ]

        # Identifier and description
        if self.metadata and self.metadata.get("description"):
            content.append(
                Text.assemble(
                    ("Description: ", "bold"),
                    self.metadata.get("description"),
                    overflow="fold",
                    end="\n\n",
                ),
            )

        # Domain section
        if self.propertyDomain:
            content.extend(
                [
                    Text("Domain:", style="bold"),
                    Panel(
                        self.propertyDomain,
                        box=rich.box.SIMPLE_HEAD,
                        padding=(0, 2),
                    ),  # Uses propertyDomain.__rich__()
                ]
            )

        return Group(*content)

    def descriptor(self) -> PropertyDescriptor:

        return PropertyDescriptor(identifier=self.identifier)


class AbstractProperty(Property):
    """Represents an Abstract Property"""

    propertyType: MeasuredPropertyTypeEnum = (
        MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )
    concretePropertyIdentifiers: list[str] | None = None
    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_descriptor(
        cls, descriptor: AbstractPropertyDescriptor
    ) -> "AbstractProperty":

        return cls(
            identifier=descriptor.identifier,
        )

    def __str__(self) -> str:
        return f"ap-{self.identifier}"

    def __eq__(self, other: object) -> bool:  # noqa: ANN401

        retval = super().__eq__(other)
        return (
            retval
            and hasattr(other, "concretePropertyIdentifiers")
            and self.concretePropertyIdentifiers == other.concretePropertyIdentifiers
        )

    def descriptor(self) -> AbstractPropertyDescriptor:

        return AbstractPropertyDescriptor(identifier=self.identifier)


class ConstitutiveProperty(Property):
    propertyType: Annotated[NonMeasuredPropertyTypeEnum, pydantic.Field()] = (
        NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE
    )

    @classmethod
    def from_descriptor(
        cls, descriptor: AbstractPropertyDescriptor
    ) -> "ConstitutiveProperty":

        return cls(
            identifier=descriptor.identifier,
        )

    def __str__(self) -> str:
        return f"cp-{self.identifier}"

    model_config = ConfigDict(frozen=True)

    def descriptor(self) -> ConstitutivePropertyDescriptor:

        return ConstitutivePropertyDescriptor(identifier=self.identifier)


class ConcreteProperty(Property):
    propertyType: Annotated[MeasuredPropertyTypeEnum, pydantic.Field()] = (
        MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )
    abstractProperty: Annotated[AbstractProperty | None, pydantic.Field()] = None
    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_descriptor(
        cls, descriptor: ConcretePropertyDescriptor
    ) -> "ConcreteProperty":

        return cls(
            identifier=descriptor.identifier,
            abstractProperty=(
                AbstractProperty.from_descriptor(descriptor.abstractProperty)
                if descriptor.abstractProperty
                else None
            ),
        )

    def __str__(self) -> str:
        return f"cp-{self.identifier}"

    def descriptor(self) -> ConcretePropertyDescriptor:
        return ConcretePropertyDescriptor(
            identifier=self.identifier,
            abstractProperty=(
                self.abstractProperty.descriptor() if self.abstractProperty else None
            ),
        )
