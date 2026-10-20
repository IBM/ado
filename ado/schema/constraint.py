# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Models for constraints on the values of properties.

A constraint names a requirement, optionally gated by a condition. The
requirement and the condition are predicates. A predicate is a comparison
of two operands, or a combination of predicates using allOf, anyOf, or not.
"""

import enum
from typing import Annotated, Self

import pydantic
from pydantic import ConfigDict

ScalarLiteral = (
    pydantic.StrictBool
    | pydantic.StrictInt
    | pydantic.StrictFloat
    | pydantic.StrictStr
    | None
)


class ComparisonOperator(enum.StrEnum):
    """Operator applied between the left and right operands of a comparison.

    ``in`` and ``notIn`` test whether the left operand is a member of the
    right operand. Every other operator compares two single values.
    """

    eq = "eq"
    ne = "ne"
    lt = "lt"
    le = "le"
    gt = "gt"
    ge = "ge"
    in_ = "in"
    notIn = "notIn"


_MEMBERSHIP_OPERATORS = frozenset({ComparisonOperator.in_, ComparisonOperator.notIn})


class PropertyOperand(pydantic.BaseModel):
    """An operand that reads the current value of a property."""

    property: Annotated[
        str,
        pydantic.Field(
            min_length=1,
            description="Identifier of the property whose value is compared",
        ),
    ]
    model_config = ConfigDict(extra="forbid", frozen=True)

    def property_identifiers(self) -> set[str]:
        """Return the property identifier this operand names.

        Returns:
            A set containing this operand's property identifier.
        """

        return {self.property}


class ValueOperand(pydantic.BaseModel):
    """An operand that is one literal value."""

    value: Annotated[
        ScalarLiteral,
        pydantic.Field(description="A single literal value"),
    ]
    model_config = ConfigDict(extra="forbid", frozen=True)

    def property_identifiers(self) -> set[str]:
        """Return the property identifiers this operand names.

        Returns:
            An empty set.
        """

        return set()


class ValuesOperand(pydantic.BaseModel):
    """An operand that is a set of literal values.

    Used as the right operand of ``in`` and ``notIn``.
    """

    values: Annotated[
        tuple[ScalarLiteral, ...],
        pydantic.Field(
            min_length=1,
            description="Literal values for a membership comparison",
        ),
    ]
    model_config = ConfigDict(extra="forbid", frozen=True)

    def property_identifiers(self) -> set[str]:
        """Return the property identifiers this operand names.

        Returns:
            An empty set.
        """

        return set()


def _operand_tag(operand: dict[str, object]) -> str:
    """Return the operand tag for a mapping, requiring exactly one operand field.

    Args:
        operand: A mapping of operand fields.

    Returns:
        The tag ``property``, ``value``, or ``values``.

    Raises:
        ValueError: If the mapping does not set exactly one of those fields.
    """

    present = [key for key in ("property", "value", "values") if key in operand]
    if len(present) == 1:
        return present[0]
    raise ValueError(
        "A comparison operand must set exactly one of property, value, or values"
    )


def _require_single_operand_field(
    operand: PropertyOperand | ValueOperand | ValuesOperand | dict[str, object],
) -> PropertyOperand | ValueOperand | ValuesOperand | dict[str, object]:
    """Reject mappings that do not set exactly one operand field.

    Args:
        operand: An operand model or a mapping of its fields.

    Returns:
        The operand unchanged when it is a model or a single-field mapping.

    Raises:
        ValueError: If a mapping does not set exactly one operand field.
    """

    if isinstance(operand, dict):
        _operand_tag(operand)
    return operand


def _comparison_operand_discriminator(
    operand: PropertyOperand | ValueOperand | ValuesOperand | dict[str, object],
) -> str:
    """Return the operand variant tag for a value.

    Args:
        operand: An operand model or a mapping of its fields.

    Returns:
        The tag ``property``, ``value``, or ``values``.

    Raises:
        ValueError: If the operand does not set exactly one of those fields.
    """

    if isinstance(operand, PropertyOperand):
        return "property"
    if isinstance(operand, ValueOperand):
        return "value"
    if isinstance(operand, ValuesOperand):
        return "values"
    if isinstance(operand, dict):
        return _operand_tag(operand)
    raise ValueError(f"Unable to determine comparison operand type for {operand!r}")


ComparisonOperand = Annotated[
    Annotated[PropertyOperand, pydantic.Tag("property")]
    | Annotated[ValueOperand, pydantic.Tag("value")]
    | Annotated[ValuesOperand, pydantic.Tag("values")],
    pydantic.Discriminator(_comparison_operand_discriminator),
    pydantic.BeforeValidator(_require_single_operand_field),
]


class Comparison(pydantic.BaseModel):
    """A comparison of a left operand to a right operand.

    At least one operand names a property. Both may name properties, in
    which case the current values of those properties are compared.
    """

    left: Annotated[
        ComparisonOperand,
        pydantic.Field(description="The left-hand operand"),
    ]
    operator: Annotated[
        ComparisonOperator,
        pydantic.Field(description="How the left operand is compared to the right"),
    ]
    right: Annotated[
        ComparisonOperand,
        pydantic.Field(description="The right-hand operand"),
    ]
    model_config = ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="after")
    def validate_operands(self) -> Self:
        """Check that the operands match the operator.

        Returns:
            This comparison.

        Raises:
            ValueError: If neither operand names a property, or the operand
                kinds do not match the operator.
        """

        if not isinstance(self.left, PropertyOperand) and not isinstance(
            self.right, PropertyOperand
        ):
            raise ValueError(
                "A comparison must reference a property on the left, the right, or both"
            )

        if self.operator in _MEMBERSHIP_OPERATORS:
            if isinstance(self.left, ValuesOperand):
                raise ValueError(
                    f"The left operand of {self.operator.value} is the value being tested. "
                    "Put the set on the right."
                )
            if isinstance(self.right, ValueOperand):
                raise ValueError(
                    f"The right operand of {self.operator.value} is the set. "
                    "Use values for a literal set."
                )
            return self

        if isinstance(self.left, ValuesOperand) or isinstance(
            self.right, ValuesOperand
        ):
            raise ValueError(
                f"Operator {self.operator.value} compares single values. "
                "Use value for a literal."
            )
        return self

    def property_identifiers(self) -> set[str]:
        """Return the property identifiers this comparison references.

        Returns:
            Identifiers named by the left and right operands.
        """

        return self.left.property_identifiers() | self.right.property_identifiers()


class AllOf(pydantic.BaseModel):
    """A predicate that holds when every nested predicate holds."""

    allOf: Annotated[
        tuple["Predicate", ...],
        pydantic.Field(
            min_length=1,
            description="Predicates that must all hold",
        ),
    ]
    model_config = ConfigDict(extra="forbid", frozen=True)

    def property_identifiers(self) -> set[str]:
        """Return the property identifiers referenced by nested predicates.

        Returns:
            The union of identifiers referenced by each predicate in allOf.
        """

        identifiers: set[str] = set()
        for predicate in self.allOf:
            identifiers.update(predicate.property_identifiers())
        return identifiers


class AnyOf(pydantic.BaseModel):
    """A predicate that holds when at least one nested predicate holds."""

    anyOf: Annotated[
        tuple["Predicate", ...],
        pydantic.Field(
            min_length=1,
            description="Predicates of which at least one must hold",
        ),
    ]
    model_config = ConfigDict(extra="forbid", frozen=True)

    def property_identifiers(self) -> set[str]:
        """Return the property identifiers referenced by nested predicates.

        Returns:
            The union of identifiers referenced by each predicate in anyOf.
        """

        identifiers: set[str] = set()
        for predicate in self.anyOf:
            identifiers.update(predicate.property_identifiers())
        return identifiers


class Not(pydantic.BaseModel):
    """A predicate that holds when the nested predicate does not hold."""

    not_: Annotated[
        "Predicate",
        pydantic.Field(
            alias="not",
            description="Predicate that must not hold",
        ),
    ]
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    def property_identifiers(self) -> set[str]:
        """Return the property identifiers referenced by the nested predicate.

        Returns:
            Identifiers referenced by the negated predicate.
        """

        return self.not_.property_identifiers()


def _predicate_tag(predicate: dict[str, object]) -> str:
    """Return the predicate tag for a mapping.

    Args:
        predicate: A mapping of predicate fields.

    Returns:
        The tag ``comparison``, ``allOf``, ``anyOf``, or ``not``.

    Raises:
        ValueError: If the mapping mixes predicate forms or is not a predicate.
    """

    combinators = [key for key in ("allOf", "anyOf", "not") if key in predicate]
    comparison_keys = [key for key in ("left", "operator", "right") if key in predicate]
    if combinators:
        if len(combinators) != 1 or comparison_keys:
            raise ValueError(
                "A predicate must be a comparison or exactly one of allOf, anyOf, or not"
            )
        return combinators[0]
    if comparison_keys:
        return "comparison"
    raise ValueError(f"Unable to determine predicate type for {predicate!r}")


def _require_single_predicate_form(
    predicate: Comparison | AllOf | AnyOf | Not | dict[str, object],
) -> Comparison | AllOf | AnyOf | Not | dict[str, object]:
    """Reject mappings that mix predicate forms.

    Args:
        predicate: A predicate model or a mapping of its fields.

    Returns:
        The predicate unchanged when it is a model or a single predicate form.

    Raises:
        ValueError: If a mapping mixes predicate forms or is not a predicate.
    """

    if isinstance(predicate, dict):
        _predicate_tag(predicate)
    return predicate


def _predicate_discriminator(
    predicate: Comparison | AllOf | AnyOf | Not | dict[str, object],
) -> str:
    """Return the predicate variant tag for a value.

    Args:
        predicate: A predicate model or a mapping of its fields.

    Returns:
        The tag ``comparison``, ``allOf``, ``anyOf``, or ``not``.

    Raises:
        ValueError: If the value is not a predicate or mixes predicate forms.
    """

    if isinstance(predicate, Comparison):
        return "comparison"
    if isinstance(predicate, AllOf):
        return "allOf"
    if isinstance(predicate, AnyOf):
        return "anyOf"
    if isinstance(predicate, Not):
        return "not"
    if isinstance(predicate, dict):
        return _predicate_tag(predicate)
    raise ValueError(f"Unable to determine predicate type for {predicate!r}")


Predicate = Annotated[
    Annotated[Comparison, pydantic.Tag("comparison")]
    | Annotated[AllOf, pydantic.Tag("allOf")]
    | Annotated[AnyOf, pydantic.Tag("anyOf")]
    | Annotated[Not, pydantic.Tag("not")],
    pydantic.Discriminator(_predicate_discriminator),
    pydantic.BeforeValidator(_require_single_predicate_form),
]

# Resolve the recursive predicate references now that Predicate is defined.
AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


class PropertyConstraint(pydantic.BaseModel):
    """A named requirement on property values, optionally under a condition.

    ``require`` must hold wherever ``when`` holds. Omit ``when`` to enforce
    ``require`` at every point.
    """

    name: Annotated[
        str,
        pydantic.Field(min_length=1, description="Name of this constraint"),
    ]
    when: Annotated[
        Predicate | None,
        pydantic.Field(
            description=(
                "Gates the constraint. require is enforced only where when is true. "
                "Omit to enforce require at every point."
            ),
        ),
    ] = None
    require: Annotated[
        Predicate,
        pydantic.Field(description="Predicate that must hold wherever when holds"),
    ]
    model_config = ConfigDict(extra="forbid", frozen=True)

    def property_identifiers(self) -> set[str]:
        """Return the property identifiers this constraint references.

        Returns:
            Identifiers referenced by require and, when set, by when.
        """

        identifiers = self.require.property_identifiers()
        if self.when is not None:
            identifiers |= self.when.property_identifiers()
        return identifiers
