# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.schema.observed_property import ObservedProperty
from orchestrator.schema.property import AbstractProperty, ConcreteProperty


def test_observed_property_hashable(experiment_reference):

    ap = AbstractProperty(identifier="test")
    op = ObservedProperty(targetProperty=ap, experimentReference=experiment_reference)
    d = {op: "some_key"}
    assert d


def test_property_equivalence_non_property(requiredProperties):
    """Test the property equivalence works"""

    # non-equivalence to non-Property subclass is determined by missing attributes identifier and propertyDomain
    for p in requiredProperties:
        assert p == p
        assert p != "somestring", "Property evaluated equivalent to random string"
        assert p != 3, "Property evaluated equivalent to integer"


def test_abstract_property_identifier_and_string_representation(
    target_property_list, abstract_properties
):

    for t, p in zip(target_property_list, abstract_properties):
        assert p.identifier == t
        assert str(p) == f"ap-{t}"

        concrete = ConcreteProperty(identifier="test", abstractProperty=p)
        assert concrete.identifier == "test"
        assert str(concrete) == "cp-test"


def test_constitutive_property_identifier_and_string_representation(
    constitutive_property_list, constitutive_properties
):
    for t, p in zip(constitutive_property_list, constitutive_properties):
        assert p.identifier == t
        assert str(p) == f"cp-{t}"
