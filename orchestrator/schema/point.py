# Copyright (c) IBM Corporation
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

    def to_entity(self) -> Entity:

        return Entity(
            constitutive_property_values=tuple(
                [
                    ConstitutivePropertyValue(
                        value=v, property=ConstitutivePropertyDescriptor(identifier=k)
                    )
                    for k, v in self.entity.items()
                ]
            )
        )
