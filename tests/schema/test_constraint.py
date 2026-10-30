# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pydantic
import pytest
import yaml

from ado.schema.constraint import (
    AllOf,
    AnyOf,
    Comparison,
    ComparisonOperator,
    Not,
    PropertyConstraint,
    PropertyOperand,
    ValueOperand,
    ValuesOperand,
)
from ado.utilities.output import pydantic_model_as_yaml

_AT_LEAST_ONE_SAMPLE_PER_GPU = {
    "name": "at_least_one_sample_per_gpu",
    "require": {
        "left": {"property": "effective_batch_size"},
        "operator": "ge",
        "right": {"property": "num_gpus"},
    },
}

_LARGE_MODEL_NEEDS_MULTIPLE_GPUS = {
    "name": "large_model_needs_multiple_gpus",
    "when": {
        "left": {"property": "model"},
        "operator": "eq",
        "right": {"value": "llama-70b"},
    },
    "require": {
        "left": {"property": "num_gpus"},
        "operator": "ge",
        "right": {"value": 4},
    },
}

_FLASH_ATTENTION_UNSUPPORTED = {
    "name": "flash_attention_unsupported_on_older_gpus",
    "when": {
        "left": {"property": "use_flash_attention"},
        "operator": "eq",
        "right": {"value": True},
    },
    "require": {
        "left": {"property": "gpu_type"},
        "operator": "notIn",
        "right": {"values": ["v100", "t4"]},
    },
}

_ADAFACTOR_UNSTABLE = {
    "name": "adafactor_unstable_at_fp16_with_flash_att",
    "when": {
        "allOf": [
            {
                "left": {"property": "precision"},
                "operator": "eq",
                "right": {"value": "fp16"},
            },
            {
                "left": {"property": "use_flash_attention"},
                "operator": "eq",
                "right": {"value": True},
            },
        ]
    },
    "require": {
        "left": {"property": "optimizer"},
        "operator": "ne",
        "right": {"value": "adafactor"},
    },
}

_LARGE_DATASET_NEEDS_CAPACITY = {
    "name": "large_dataset_needs_capacity_or_low_precision",
    "when": {
        "left": {"property": "dataset"},
        "operator": "eq",
        "right": {"value": "large"},
    },
    "require": {
        "anyOf": [
            {
                "left": {"property": "num_gpus"},
                "operator": "ge",
                "right": {"value": 8},
            },
            {
                "left": {"property": "precision"},
                "operator": "eq",
                "right": {"value": "fp8"},
            },
        ]
    },
}

_WARMUP_ONLY_FOR_COSINE = {
    "name": "warmup_only_meaningful_for_cosine",
    "when": {
        "not": {
            "left": {"property": "lr_schedule"},
            "operator": "eq",
            "right": {"value": "cosine"},
        }
    },
    "require": {
        "left": {"property": "warmup_steps"},
        "operator": "eq",
        "right": {"value": 0},
    },
}

_LARGE_MODEL_AS_REQUIRE_ONLY = {
    "name": "large_model_needs_multiple_gpus",
    "require": {
        "anyOf": [
            {
                "not": {
                    "left": {"property": "model"},
                    "operator": "eq",
                    "right": {"value": "llama-70b"},
                }
            },
            {
                "left": {"property": "num_gpus"},
                "operator": "ge",
                "right": {"value": 4},
            },
        ]
    },
}

_ISSUE_EXAMPLES = [
    _AT_LEAST_ONE_SAMPLE_PER_GPU,
    _LARGE_MODEL_NEEDS_MULTIPLE_GPUS,
    _FLASH_ATTENTION_UNSUPPORTED,
    _ADAFACTOR_UNSTABLE,
    _LARGE_DATASET_NEEDS_CAPACITY,
    _WARMUP_ONLY_FOR_COSINE,
    _LARGE_MODEL_AS_REQUIRE_ONLY,
]


def _assert_round_trip(model: pydantic.BaseModel) -> None:
    """Create a model from its python and JSON dumps and require equality."""

    restored = type(model).model_validate(model.model_dump())
    assert restored == model
    restored_from_json = type(model).model_validate(model.model_dump(mode="json"))
    assert restored_from_json == model


@pytest.mark.parametrize(
    "payload",
    _ISSUE_EXAMPLES,
    ids=[
        "at_least_one_sample_per_gpu",
        "large_model_needs_multiple_gpus",
        "flash_attention_unsupported_on_older_gpus",
        "adafactor_unstable_at_fp16_with_flash_att",
        "large_dataset_needs_capacity_or_low_precision",
        "warmup_only_meaningful_for_cosine",
        "large_model_needs_multiple_gpus_require_only",
    ],
)
def test_property_constraint_examples_round_trip(payload: dict) -> None:
    """Test the constraint shapes from the feature discussion survive dump and load."""

    constraint = PropertyConstraint.model_validate(payload)
    _assert_round_trip(constraint)
    assert (
        constraint.model_validate(yaml.safe_load(pydantic_model_as_yaml(constraint)))
        == constraint
    )


def test_construct_constraint_from_objects() -> None:
    """Test a constraint built from model instances dumps and loads."""

    constraint = PropertyConstraint(
        name="large_model_needs_multiple_gpus",
        when=Comparison(
            left=PropertyOperand(property="model"),
            operator=ComparisonOperator.eq,
            right=ValueOperand(value="llama-70b"),
        ),
        require=Comparison(
            left=PropertyOperand(property="num_gpus"),
            operator=ComparisonOperator.ge,
            right=ValueOperand(value=4),
        ),
    )
    _assert_round_trip(constraint)
    assert constraint.when is not None
    assert constraint.property_identifiers() == {"model", "num_gpus"}


def test_when_defaults_to_none() -> None:
    """Test omitting when means the requirement is unconditional."""

    constraint = PropertyConstraint.model_validate(_AT_LEAST_ONE_SAMPLE_PER_GPU)
    assert constraint.when is None
    assert "when" not in constraint.model_dump(exclude_none=True)


def test_bool_literal_is_preserved() -> None:
    """Test a boolean literal stays a bool rather than becoming an int."""

    constraint = PropertyConstraint.model_validate(_FLASH_ATTENTION_UNSUPPORTED)
    assert isinstance(constraint.when, Comparison)
    assert isinstance(constraint.when.right, ValueOperand)
    assert constraint.when.right.value is True
    assert type(constraint.when.right.value) is bool


def test_membership_values_stay_a_sequence_of_strings() -> None:
    """Test notIn keeps the literal set it was given."""

    constraint = PropertyConstraint.model_validate(_FLASH_ATTENTION_UNSUPPORTED)
    assert isinstance(constraint.require, Comparison)
    assert isinstance(constraint.require.right, ValuesOperand)
    assert constraint.require.right.values == ("v100", "t4")
    assert constraint.require.operator is ComparisonOperator.notIn


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        (ComparisonOperator.lt, 1.5),
        (ComparisonOperator.le, 0),
        (ComparisonOperator.gt, 2),
        (ComparisonOperator.eq, None),
        (ComparisonOperator.ne, "fp16"),
    ],
)
def test_single_value_operators_round_trip(
    operator: ComparisonOperator,
    value: float | int | str | None,  # noqa: PYI041
) -> None:
    """Test ordering and equality comparisons against a literal value."""

    comparison = Comparison(
        left=PropertyOperand(property="x"),
        operator=operator,
        right=ValueOperand(value=value),
    )
    _assert_round_trip(comparison)
    assert isinstance(comparison.right, ValueOperand)
    assert comparison.right.value == value
    assert type(comparison.right.value) is type(value)


def test_in_operator_with_literal_set() -> None:
    """Test membership of a property in a literal set."""

    comparison = Comparison.model_validate(
        {
            "left": {"property": "model"},
            "operator": "in",
            "right": {"values": ["llama-70b", "qwen-28b"]},
        }
    )
    _assert_round_trip(comparison)
    assert comparison.operator is ComparisonOperator.in_
    assert comparison.property_identifiers() == {"model"}


def test_literal_on_the_left_and_property_on_the_right() -> None:
    """Test a comparison that places the literal on the left."""

    comparison = Comparison(
        left=ValueOperand(value=4),
        operator=ComparisonOperator.le,
        right=PropertyOperand(property="num_gpus"),
    )
    _assert_round_trip(comparison)
    assert comparison.property_identifiers() == {"num_gpus"}


def test_property_to_property_membership() -> None:
    """Test in when both operands are properties."""

    comparison = Comparison(
        left=PropertyOperand(property="gpu_type"),
        operator=ComparisonOperator.in_,
        right=PropertyOperand(property="allowed_gpus"),
    )
    _assert_round_trip(comparison)
    assert comparison.property_identifiers() == {"gpu_type", "allowed_gpus"}


def test_nested_predicates_round_trip() -> None:
    """Test allOf containing not containing anyOf."""

    predicate = AllOf(
        allOf=(
            Not(
                not_=AnyOf(
                    anyOf=(
                        Comparison(
                            left=PropertyOperand(property="precision"),
                            operator=ComparisonOperator.eq,
                            right=ValueOperand(value="fp16"),
                        ),
                        Comparison(
                            left=PropertyOperand(property="num_gpus"),
                            operator=ComparisonOperator.ge,
                            right=ValueOperand(value=4),
                        ),
                    )
                )
            ),
            Comparison(
                left=PropertyOperand(property="optimizer"),
                operator=ComparisonOperator.ne,
                right=ValueOperand(value="adafactor"),
            ),
        )
    )
    _assert_round_trip(predicate)
    assert predicate.property_identifiers() == {"precision", "num_gpus", "optimizer"}
    dumped = predicate.model_dump(mode="json")
    assert "not" in dumped["allOf"][0]
    assert "not_" not in dumped["allOf"][0]


def test_property_identifiers_from_nested_issue_examples() -> None:
    """Test property identifiers are collected from when and require."""

    adafactor = PropertyConstraint.model_validate(_ADAFACTOR_UNSTABLE)
    assert adafactor.property_identifiers() == {
        "precision",
        "use_flash_attention",
        "optimizer",
    }

    warmup = PropertyConstraint.model_validate(_WARMUP_ONLY_FOR_COSINE)
    assert warmup.property_identifiers() == {"lr_schedule", "warmup_steps"}

    rewritten = PropertyConstraint.model_validate(_LARGE_MODEL_AS_REQUIRE_ONLY)
    assert rewritten.property_identifiers() == {"model", "num_gpus"}


def test_not_serializes_with_not_key() -> None:
    """Test the negation predicate uses the key not."""

    constraint = PropertyConstraint.model_validate(_WARMUP_ONLY_FOR_COSINE)
    dumped = constraint.model_dump(mode="json", exclude_none=True)
    assert set(dumped["when"]) == {"not"}
    assert dumped["when"]["not"]["left"] == {"property": "lr_schedule"}


def test_list_of_constraints_round_trip() -> None:
    """Test a sequence of constraints dumps and loads."""

    constraints = [
        PropertyConstraint.model_validate(payload) for payload in _ISSUE_EXAMPLES
    ]
    adapter = pydantic.TypeAdapter(list[PropertyConstraint])
    restored = adapter.validate_python(adapter.dump_python(constraints))
    assert restored == constraints


def test_json_schema_builds() -> None:
    """Test the recursive constraint models expose a JSON schema."""

    schema = PropertyConstraint.model_json_schema()
    assert "properties" in schema
    assert "require" in schema["properties"]


def test_comparison_requires_a_property() -> None:
    """Test a comparison of two literals is rejected."""

    with pytest.raises(
        pydantic.ValidationError,
        match="At least one side of a comparison must be a property",
    ):
        Comparison.model_validate(
            {
                "left": {"value": 1},
                "operator": "lt",
                "right": {"value": 2},
            }
        )


def test_membership_rejects_a_scalar_set() -> None:
    """Test in requires a values operand for a literal set."""

    with pytest.raises(pydantic.ValidationError, match="literal set"):
        Comparison.model_validate(
            {
                "left": {"property": "model"},
                "operator": "in",
                "right": {"value": "llama-70b"},
            }
        )


def test_membership_rejects_a_set_on_the_left() -> None:
    """Test the value being tested sits on the left of in."""

    with pytest.raises(pydantic.ValidationError, match="left operand"):
        Comparison.model_validate(
            {
                "left": {"values": ["a", "b"]},
                "operator": "notIn",
                "right": {"property": "gpu_type"},
            }
        )


def test_single_value_operator_rejects_values() -> None:
    """Test equality and ordering operators reject a values operand."""

    with pytest.raises(pydantic.ValidationError, match="single values"):
        Comparison.model_validate(
            {
                "left": {"property": "model"},
                "operator": "eq",
                "right": {"values": ["llama-70b"]},
            }
        )


def test_value_rejects_a_list() -> None:
    """Test a list literal belongs in values."""

    with pytest.raises(pydantic.ValidationError):
        ValueOperand.model_validate({"value": ["llama-70b", "qwen-28b"]})


def test_empty_values_rejected() -> None:
    """Test a membership set contains at least one value."""

    with pytest.raises(pydantic.ValidationError):
        ValuesOperand(values=())


def test_empty_all_of_rejected() -> None:
    """Test allOf contains at least one predicate."""

    with pytest.raises(pydantic.ValidationError):
        AllOf.model_validate({"allOf": []})


def test_empty_any_of_rejected() -> None:
    """Test anyOf contains at least one predicate."""

    with pytest.raises(pydantic.ValidationError):
        AnyOf.model_validate({"anyOf": []})


def test_operand_rejects_multiple_fields() -> None:
    """Test an operand sets exactly one of property, value, or values."""

    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        Comparison.model_validate(
            {
                "left": {"property": "model", "value": "llama-70b"},
                "operator": "eq",
                "right": {"property": "num_gpus"},
            }
        )


def test_operand_rejects_an_empty_object() -> None:
    """Test an operand names a property or a literal."""

    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        Comparison.model_validate(
            {
                "left": {},
                "operator": "eq",
                "right": {"property": "model"},
            }
        )


def test_predicate_rejects_mixed_forms() -> None:
    """Test a predicate is one comparison or one combinator."""

    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        AllOf.model_validate(
            {
                "allOf": [
                    {
                        "allOf": [],
                        "anyOf": [],
                    }
                ]
            }
        )


def test_unknown_operator_rejected() -> None:
    """Test an operator outside the comparison set is rejected."""

    with pytest.raises(pydantic.ValidationError):
        Comparison.model_validate(
            {
                "left": {"property": "model"},
                "operator": "contains",
                "right": {"value": "llama"},
            }
        )


def test_extra_fields_rejected() -> None:
    """Test constraint models reject unknown fields."""

    with pytest.raises(pydantic.ValidationError, match="Extra inputs"):
        PropertyConstraint.model_validate(
            {
                "name": "example",
                "note": "nope",
                "require": {
                    "left": {"property": "x"},
                    "operator": "eq",
                    "right": {"value": 1},
                },
            }
        )


def test_empty_name_rejected() -> None:
    """Test a constraint name is a non-empty string."""

    with pytest.raises(pydantic.ValidationError):
        PropertyConstraint.model_validate(
            {
                "name": "",
                "require": {
                    "left": {"property": "x"},
                    "operator": "eq",
                    "right": {"value": 1},
                },
            }
        )


def test_empty_property_identifier_rejected() -> None:
    """Test a property operand names a non-empty identifier."""

    with pytest.raises(pydantic.ValidationError):
        PropertyOperand(property="")


def test_integer_is_not_coerced_to_string() -> None:
    """Test a numeric literal stays numeric."""

    comparison = Comparison(
        left=PropertyOperand(property="num_gpus"),
        operator=ComparisonOperator.eq,
        right=ValueOperand(value=4),
    )
    assert isinstance(comparison.right, ValueOperand)
    assert comparison.right.value == 4
    assert type(comparison.right.value) is int


def test_string_is_not_coerced_to_number() -> None:
    """Test a numeric string stays a string."""

    operand = ValueOperand.model_validate({"value": "4"})
    assert operand.value == "4"
    assert type(operand.value) is str
