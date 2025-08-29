# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing

import pydantic
import pytest

from orchestrator.schema.property import ConstitutiveProperty
from orchestrator.schema.property_value import (
    PropertyValue,
    ValueTypeEnum,
)


@pytest.fixture
def python_type_value_examples() -> (
    typing.Dict[typing.Type, typing.Tuple[ValueTypeEnum, typing.Any]]
):

    return {
        float: (ValueTypeEnum.NUMERIC_VALUE_TYPE, 3.0),
        int: (ValueTypeEnum.NUMERIC_VALUE_TYPE, 1),
        type(None): (ValueTypeEnum.NUMERIC_VALUE_TYPE, None),
        str: (ValueTypeEnum.STRING_VALUE_TYPE, "string"),
        list: (ValueTypeEnum.VECTOR_VALUE_TYPE, [0, "a", 10]),
        bytes: (ValueTypeEnum.BLOB_VALUE_TYPE, b"PNG\r89\n\x1a\n\x00\x00"),
    }


@pytest.fixture(params=[int, float, str, bytes, list, type(None)])
def value_example(
    python_type_value_examples, request
) -> typing.Tuple[ValueTypeEnum, typing.Any]:

    return python_type_value_examples[request.param]


@pytest.fixture(params=[int, float, str, bytes, list, type(None)])
def test_value_example(
    python_type_value_examples, request
) -> typing.Tuple[ValueTypeEnum, typing.Any]:

    return python_type_value_examples[request.param]


@pytest.fixture(params=[int, float, str, bytes, list, type(None)])
def property_value(request) -> typing.Tuple[PropertyValue, typing.Type]:
    """Returns PropertyValue instance of different types.

    Note: Does not set the valueType field explicitly in order to test that it is detected correctly
    """

    prop = ConstitutiveProperty(identifier="cons_prop")
    if request.param is int:
        val = PropertyValue(value=3, property=prop)
    elif request.param is float:
        val = PropertyValue(value=3.0, property=prop)
    elif request.param is str:
        val = PropertyValue(value="string", property=prop)
    elif request.param is bytes:
        val = PropertyValue(
            value=b"PNG\r89\n\x1a\n\x00\x00",
            property=prop,
        )
    elif request.param is list:
        val = PropertyValue(value=[0, "a", 10], property=prop)
    elif request.param is type(None):
        val = PropertyValue(value=None, property=prop)
    else:
        raise ValueError(f"Unexpected param {request.param}")

    return val, request.param


def test_property_value_preserves_value_type(
    property_value: typing.Tuple[PropertyValue, typing.Type],
):
    """Test that PropertyValue preserved the type of the value added to it

    For example that if the value is an int its not returned as a float"""

    val, value_type = property_value

    assert isinstance(
        val.value, value_type
    ), f"PropertyValue did not store a {value_type} value as {value_type}. Instead returned {type(val.value)}"


def test_property_value_preserves_value_type_after_json_serialization(
    property_value: typing.Tuple[PropertyValue, typing.Type],
):
    """Test that PropertyValue preserved the type of the value added to it after it is dumped as json and re-read

    For example that if the value is an int its not returned as a float"""

    import json

    val, value_type = property_value

    ser = val.model_dump_json()
    dser = PropertyValue.model_validate(json.loads(ser))

    assert isinstance(
        dser.value, value_type
    ), f"PropertyValue did not store a {value_type} value as {value_type}. Instead returned {type(val.value)}"


def test_property_value_preserves_value_type_after_serialization(
    property_value: typing.Tuple[PropertyValue, typing.Type],
):
    """Test that PropertyValue preserved the type of the value added to it after it is dumped and re-read

    For example that if the value is an int its not returned as a float"""

    val, value_type = property_value

    ser = val.model_dump()
    dser = PropertyValue.model_validate(ser)

    assert isinstance(
        dser.value, value_type
    ), f"PropertyValue did not store a {value_type} value as {value_type}. Instead returned {type(val.value)}"


def test_property_value_checks_value_type(value_example, test_value_example):
    """Tests if PropertyValue validates the type of the value against the valueType field"""

    prop = ConstitutiveProperty(identifier="cons_prop")
    example_type, example_value = value_example
    PropertyValue(value=example_value, property=prop, valueType=example_type)

    # Check against the other example values.
    # Values with the same type are fine, values of different types should fail
    test_type, test_value = test_value_example
    if test_type is not example_type:
        # Due to a bug, as a temp measure special behaviour has been enabled
        # Because it is temporary, we xfail it.
        # This indicates it should fail but currently is being allows
        #
        # Passing str value with NUMERIC_VALUE_TYPE will change it to STRING_VALUE_TYPE
        if type(test_value) is str and example_type is ValueTypeEnum.NUMERIC_VALUE_TYPE:
            PropertyValue(value=test_value, property=prop, valueType=example_type)
            pytest.xfail(
                "Automatically changing value type from NUMERIC_VALUE_TYPE to STRING_VALUE_TYPE to match string value."
                " This is being allowed temporarily but should fail"
            )
        elif (
            type(test_value) is list
            and example_type is ValueTypeEnum.NUMERIC_VALUE_TYPE
        ):
            PropertyValue(value=test_value, property=prop, valueType=example_type)
            pytest.xfail(
                "Automatically changing value type from NUMERIC_VALUE_TYPE to VECTOR_VALUE_TYPE to match list value."
                " This is being allowed temporarily but should fail"
            )
        elif type(test_value) is str and example_type is ValueTypeEnum.BLOB_VALUE_TYPE:
            # strings with BLOB_VALUE_TYPE are ALLOWED to be converted to bytes
            # i.e. this is not the same case as previous two
            PropertyValue(value=test_value, property=prop, valueType=example_type)
        else:
            with pytest.raises(pydantic.ValidationError):
                PropertyValue(value=test_value, property=prop, valueType=example_type)
    else:
        PropertyValue(value=test_value, property=prop, valueType=example_type)


def test_string_with_bytes_type_converted_to_blob():

    prop = ConstitutiveProperty(identifier="cons_prop")
    val = PropertyValue(
        value="PNG\r89\n\x1a\n\x00\x00",
        property=prop,
        valueType=ValueTypeEnum.BLOB_VALUE_TYPE,
    )

    assert (
        val.value == b"PNG\r89\n\x1a\n\x00\x00"
    ), "String value with type BLOB_VALUE_TYPE was not converted to bytes as expected"


def test_type_detection(property_value: typing.Tuple[PropertyValue, typing.Type]):
    """Test the value has type correct seto"""

    val, value_type = property_value
    if value_type in [int, float]:
        assert val.valueType == ValueTypeEnum.NUMERIC_VALUE_TYPE
    elif value_type is str:
        assert val.valueType == ValueTypeEnum.STRING_VALUE_TYPE
    elif value_type is list:
        assert val.valueType == ValueTypeEnum.VECTOR_VALUE_TYPE
    elif value_type is bytes:
        assert val.valueType == ValueTypeEnum.BLOB_VALUE_TYPE
    elif value_type is type(None):
        # Treating None as a Numeric type currently
        assert val.valueType == ValueTypeEnum.NUMERIC_VALUE_TYPE
    else:
        pytest.fail(f"Test of value type {value_type} with value {val} not implemented")


def test_uncertain_property_value(property_value):
    """Test the uncertain property works"""

    val, value_type = property_value
    assert val.isUncertain() is False

    uncertain_val = PropertyValue(
        uncertainty=True, **val.model_dump(exclude_defaults=True)
    )
    assert uncertain_val.isUncertain() is True
