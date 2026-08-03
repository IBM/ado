# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import datetime
import enum
import json
import uuid

import pydantic
import pytest

from ado.utilities.pydantic import pydantic_aware_json_serializer


class _Color(str, enum.Enum):
    RED = "red"


class _Inner(pydantic.BaseModel):
    value: float
    when: datetime.datetime


class _Outer(pydantic.BaseModel):
    name: str
    uid: uuid.UUID
    color: _Color
    inner: _Inner


# ---------------------------------------------------------------------------
# Pydantic model path
# ---------------------------------------------------------------------------


def test_pydantic_model_returns_json_string() -> None:
    """A Pydantic model is serialised to a JSON string."""
    model = _Inner(
        value=1.5, when=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    )
    result = pydantic_aware_json_serializer(model)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["value"] == 1.5
    assert isinstance(parsed["when"], str)


def test_pydantic_nested_model_is_fully_serialised() -> None:
    """Nested models and special types (UUID, Enum, datetime) are recursively serialised."""
    uid = uuid.uuid4()
    model = _Outer(
        name="x",
        uid=uid,
        color=_Color.RED,
        inner=_Inner(
            value=3.14,
            when=datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc),
        ),
    )
    result = pydantic_aware_json_serializer(model)
    parsed = json.loads(result)
    assert parsed["uid"] == str(uid)
    assert parsed["color"] == "red"
    assert isinstance(parsed["inner"], dict)
    assert isinstance(parsed["inner"]["when"], str)


def test_pydantic_model_round_trip() -> None:
    """Serialising then re-parsing a model produces an equal instance with correct field values."""
    original = _Inner(
        value=2.718,
        when=datetime.datetime(2024, 3, 15, 12, 0, tzinfo=datetime.timezone.utc),
    )
    result = pydantic_aware_json_serializer(original)
    recovered = _Inner.model_validate_json(result)
    assert recovered == original
    assert recovered.value == original.value
    assert recovered.when == original.when


def test_pydantic_nested_model_round_trip() -> None:
    """Serialising then re-parsing a nested model preserves UUID, Enum, datetime, and nested values."""
    uid = uuid.uuid4()
    original = _Outer(
        name="round-trip",
        uid=uid,
        color=_Color.RED,
        inner=_Inner(
            value=1.41,
            when=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        ),
    )
    result = pydantic_aware_json_serializer(original)
    recovered = _Outer.model_validate_json(result)
    assert recovered == original
    assert recovered.uid == uid
    assert recovered.color is _Color.RED
    assert recovered.inner.value == original.inner.value
    assert recovered.inner.when == original.inner.when


def test_root_model_serialised_correctly() -> None:
    """A RootModel is serialised to its unwrapped JSON value."""
    ListModel = pydantic.RootModel[list[int]]
    result = pydantic_aware_json_serializer(ListModel([1, 2, 3]))
    assert isinstance(result, str)
    assert json.loads(result) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Non-Pydantic path
# ---------------------------------------------------------------------------


def test_plain_dict_serialised_to_string() -> None:
    """A plain dict is serialised to a JSON string."""
    result = pydantic_aware_json_serializer({"k": "v", "n": 1})
    assert isinstance(result, str)
    assert json.loads(result) == {"k": "v", "n": 1}


def test_plain_list_serialised_to_string() -> None:
    """A plain list is serialised to a JSON string."""
    result = pydantic_aware_json_serializer([1, "two", 3.0])
    assert isinstance(result, str)
    assert json.loads(result) == [1, "two", 3.0]


def test_primitive_values_serialised_to_string() -> None:
    """Primitive JSON-compatible values are serialised to JSON strings."""
    for val in (None, True, False, 42, 3.14, "hello"):
        result = pydantic_aware_json_serializer(val)
        assert isinstance(result, str)
        assert json.loads(result) == val


def test_non_serialisable_raises() -> None:
    """Non-serialisable objects raise TypeError."""
    with pytest.raises(TypeError):
        pydantic_aware_json_serializer(object())
