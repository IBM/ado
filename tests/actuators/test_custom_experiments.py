# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import re
from typing import Literal

import pytest

from orchestrator.modules.actuators import custom_experiments
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.point import SpacePoint
from orchestrator.schema.property import ConstitutiveProperty


def test_custom_experiment_unknown_keys_are_dropped() -> None:
    @custom_experiments.custom_experiment(output_property_identifiers=["x", "y"])
    def f(x: int, y: int):
        return {"x": 1, "y": 2, "z": 3, "foo": 4}

    exp = f._experiment
    entity = SpacePoint.model_validate({"entity": {"x": 10, "y": 20}}).to_entity()
    observed_values = custom_experiments._call_decorated_custom_experiment(
        f, exp, entity
    )
    identifiers = {v.property.targetProperty.identifier for v in observed_values}
    assert identifiers == {"x", "y"}


def test_custom_experiment_only_unknown_keys_raises_value_error() -> None:
    @custom_experiments.custom_experiment(output_property_identifiers=["x", "y"])
    def f(x: int, y: int):
        return {"z": 3, "foo": 4}  # none of these match the output property identifiers

    exp = f._experiment
    entity = SpacePoint.model_validate({"entity": {"x": 1, "y": 2}}).to_entity()
    with pytest.raises(ValueError, match="No valid output properties"):
        custom_experiments._call_decorated_custom_experiment(f, exp, entity)


def test_custom_experiment_partial_output_keys() -> None:
    @custom_experiments.custom_experiment(output_property_identifiers=["a", "b", "c"])
    def f(a: int, b: int, c: int):
        return {"a": 10, "junk": 999}

    exp = f._experiment
    entity = SpacePoint.model_validate({"entity": {"a": 1, "b": 2, "c": 3}}).to_entity()
    observed_values = custom_experiments._call_decorated_custom_experiment(
        f, exp, entity
    )
    identifiers = {v.property.targetProperty.identifier for v in observed_values}
    assert "a" in identifiers
    assert "junk" not in identifiers


def test_custom_experiment_exact_output_keys() -> None:
    @custom_experiments.custom_experiment(output_property_identifiers=["foo", "bar"])
    def f(foo: int, bar: int):
        return {"foo": 123, "bar": 456}

    exp = f._experiment
    entity = SpacePoint.model_validate({"entity": {"foo": 1, "bar": 2}}).to_entity()
    observed_values = custom_experiments._call_decorated_custom_experiment(
        f, exp, entity
    )
    identifiers = {v.property.targetProperty.identifier for v in observed_values}
    assert "foo" in identifiers
    assert "bar" in identifiers
    assert len(observed_values) == 2


def test_infer_domain_and_property_type() -> None:
    """Tests that given a parameter name, its type and a default the correct behaviour is observed"""

    fn = custom_experiments._infer_domain_and_property
    # int
    p = fn("a", int, 42)
    assert p.propertyDomain.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    # float
    p = fn("b", float, 3.1)
    assert p.propertyDomain.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
    # bool
    p = fn("c", bool, True)
    assert p.propertyDomain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE
    # str
    p = fn("d", str, "hello")
    assert (
        p.propertyDomain.variableType == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
    )
    assert p.propertyDomain.values == ["hello"]
    # Literal
    p = fn("e", Literal["X", "Y"], "X")
    assert p.propertyDomain.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
    assert set(p.propertyDomain.values) == {"X", "Y"}
    # bytes is not supported
    with pytest.raises(ValueError, match=r"Unsupported annotation: <class 'bytes'>"):
        _ = fn("f", bytes, b"err")


def test_derive_required_properties_from_signature_basic() -> None:
    def f(a: int, b: float, c: int = 1) -> None:
        pass

    result = custom_experiments.derive_required_properties_from_signature(
        f, optional_property_identifiers=[]
    )
    # a, b expected (no-domain), c skipped as optional
    ids = {r.identifier for r in result}
    assert ids == {"a", "b"}

    # Check that missing annotation raises error
    def f(a, b: float, c: int = 1) -> None:
        pass

    with pytest.raises(
        ValueError, match=r"Unsupported annotation: <class 'inspect._empty'>"
    ):
        custom_experiments.derive_required_properties_from_signature(
            f, optional_property_identifiers=[]
        )


def test_get_parameterization_success_and_failure() -> None:
    import inspect

    from orchestrator.schema.property import ConstitutiveProperty

    def g(x=7, y=9) -> None:
        pass

    sig = inspect.signature(g)
    ps = [ConstitutiveProperty(identifier="x"), ConstitutiveProperty(identifier="y")]
    paramz = custom_experiments.get_parameterization(ps, sig)
    assert paramz["x"] == 7
    assert paramz["y"] == 9

    def g2(x) -> None:
        pass

    sig2 = inspect.signature(g2)
    with pytest.raises(
        ValueError, match=re.escape("Parameterization missing for: ['x']")
    ):
        custom_experiments.get_parameterization(
            [ConstitutiveProperty(identifier="x")], sig2
        )


def test_derive_optional_properties_and_parameterization_basic_types_and_unsupported() -> (
    None
):
    # covers int, float, bool, str, literal, and unsupported

    def fn(
        i: int = 1,
        f: float = 2.0,
        b: bool = False,
        s: str = "abc",
        lit: Literal["A", "B"] = "A",
    ) -> None:
        pass

    optionals, _ = custom_experiments.derive_optional_properties_and_parameterization(
        fn, []
    )
    types = {p.identifier: p.propertyDomain.variableType for p in optionals}
    print(types)
    assert types["i"] == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    assert types["f"] == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
    assert types["b"] == VariableTypeEnum.BINARY_VARIABLE_TYPE
    assert types["s"] == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
    assert types["lit"] == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE

    # bytes is not supported for inference - check fails
    def fn(
        i: int = 1,
        f: float = 2.0,
        b: bool = False,
        s: str = "abc",
        lit: Literal["A", "B"] = "A",
        n: bytes = b"foo",
    ) -> None:
        pass

    with pytest.raises(ValueError, match="Unsupported annotation: <class 'bytes'>"):
        optionals, _ = (
            custom_experiments.derive_optional_properties_and_parameterization(fn, [])
        )

    # i has no annotation - check fails
    def fn(
        i=1,
        f: float = 2.0,
        b: bool = False,
        s: str = "abc",
        lit: Literal["A", "B"] = "A",
    ) -> None:
        pass

    with pytest.raises(
        ValueError, match=r"Unsupported annotation: <class 'inspect._empty'>"
    ):
        optionals, _ = (
            custom_experiments.derive_optional_properties_and_parameterization(fn, [])
        )


def test_check_parameters_and_infer() -> None:

    def fn(a: int, b: float, c: int = 1) -> None:
        pass

    optionals, parameterization, required_properties = (
        custom_experiments.check_parameters_and_infer(fn, None, None, None)
    )

    assert len(optionals) == 1
    assert optionals[0].identifier == "c"
    assert (
        optionals[0].propertyDomain.variableType
        == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    )
    assert optionals[0].propertyDomain.interval == 1

    assert len(parameterization) == 1
    assert parameterization["c"] == 1

    assert len(required_properties) == 2
    assert required_properties[0].identifier == "a"
    assert (
        required_properties[0].propertyDomain.variableType
        == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    )
    assert required_properties[0].propertyDomain.interval == 1
    assert required_properties[1].identifier == "b"
    assert (
        required_properties[1].propertyDomain.variableType
        == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
    )
    assert required_properties[1].propertyDomain.interval is None

    # Check if we pass in optional properties the parameterization is derived from the function signature
    # and the optional property returned is the same as the one passed in
    optionals, parameterization, required_properties = (
        custom_experiments.check_parameters_and_infer(
            func=fn,
            _required_properties=None,
            _optional_properties=[
                ConstitutiveProperty(
                    identifier="c",
                    propertyDomain=PropertyDomain(
                        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
                        interval=2,
                        domainRange=[0, 10],
                    ),
                )
            ],
            _parameterization=None,
        )
    )
    assert len(optionals) == 1
    assert optionals[0].propertyDomain == PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        interval=2,
        domainRange=[0, 10],
    )
    assert parameterization["c"] == 1
