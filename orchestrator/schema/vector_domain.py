# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import itertools

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum


class VectorPropertyDomain(BaseModel):
    element_domain: PropertyDomain = Field(..., description="Domain of elements")
    number_elements: int = Field(..., description="Length/dimension of the vector")
    variableType: VariableTypeEnum = Field(
        default=VariableTypeEnum.VECTOR_VARIABLE_TYPE
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("variableType")
    def variableType_matches_values(cls, value, values: "pydantic.FieldValidationInfo"):
        if value != VariableTypeEnum.VECTOR_VARIABLE_TYPE:
            raise ValueError("VariableType must be VECTOR_VARIABLE_TYPE")
        return value

    def valueInDomain(self, value: list) -> bool:
        """Check that all elements in the vector are in the element_domain."""
        if not isinstance(value, (list, tuple)) or len(value) != self.number_elements:
            return False

        return all(self.element_domain.valueInDomain(v) for v in value)

    def isSubDomain(self, otherDomain: "VectorPropertyDomain") -> bool:
        """Must be a subdomain only to another VectorPropertyDomain."""
        # Must be a VectorPropertyDomain and have the proper variableType (robustness)
        if not hasattr(otherDomain, "variableType") or (
            otherDomain.variableType != self.variableType
        ):
            return False
        # Must have equal or fewer dimensions
        if self.number_elements > otherDomain.number_elements:
            return False
        # Each element subdomain
        return self.element_domain.isSubDomain(otherDomain.element_domain)

    @property
    def domain_values(self) -> list:
        # The cartesian product of the element domain values, number_elements times
        # Returns a list of vectors
        try:
            elem_values = self.element_domain.domain_values
        except Exception as e:
            raise ValueError(
                f"element_domain must be discrete and have domain_values: {e!s}"
            )
        # Cartesian product
        return list(itertools.product(elem_values, repeat=self.number_elements))

    @property
    def size(self) -> int:
        """Returns the size (number of possible vectors) if countable."""

        n_elem_values = self.element_domain.size
        return n_elem_values**self.number_elements

    def __eq__(self, other):
        if not isinstance(other, VectorPropertyDomain):
            return False
        return (
            self.number_elements == other.number_elements
            and self.element_domain == other.element_domain
            and self.variableType == other.variableType
        )

    def _repr_pretty_(self, p, cycle=False):
        if cycle:
            p.text("Cycle detected")
        else:
            p.text(f"Type: {self.variableType}")
            p.breakable()
            p.text(f"Number of elements: {self.number_elements}")
            p.breakable()
            with p.group(2, "Element Domain:"):
                p.break_()
                p.pretty(self.element_domain)
