# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import pydantic

from orchestrator.schema.entity import Entity
from orchestrator.schema.property import ConstitutivePropertyDescriptor
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference


class SpacePoint(pydantic.BaseModel):
    """A simplified representation of an Entity and an associated set of experiments"""

    entity: Annotated[
        dict[str, typing.Any] | None,
        pydantic.Field(description="A dictionary of property name:value pairs"),
    ] = None
    experiments: Annotated[
        list[ExperimentReference] | None,
        pydantic.Field(description="A list of experiments"),
    ] = None

    def to_entity(self, generatorid: str = "unk") -> Entity:
        """Convert SpacePoint to Entity.

        Args:
            generatorid: Identifier for the generator that created this entity.
                        Defaults to "unk" if not specified.

        Returns:
            Entity with the specified generatorid and constitutive property values.
        """
        return Entity(
            generatorid=generatorid,
            constitutive_property_values=tuple(
                [
                    ConstitutivePropertyValue(
                        value=v, property=ConstitutivePropertyDescriptor(identifier=k)
                    )
                    for k, v in self.entity.items()
                ]
            ),
        )
