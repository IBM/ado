# Regression tests for issue #1632: PropertyDomain.isSubDomain raises TypeError for discrete domains without a domainRange
import math
import pytest

from ado.schema.domain import PropertyDomain, VariableTypeEnum


def test_case1_values_against_unbounded_interval():
    """P(values=[1]).isSubDomain(P(interval=1)) should return True."""
    tested = PropertyDomain(values=[1])
    enclosing = PropertyDomain(interval=1)
    assert tested.isSubDomain(enclosing) is True


def test_case2_unbounded_against_values():
    """P(interval=1).isSubDomain(P(values=[1, 2])) should return False."""
    tested = PropertyDomain(interval=1)
    enclosing = PropertyDomain(values=[1, 2])
    assert tested.isSubDomain(enclosing) is False


def test_case3_binary_against_unbounded_interval():
    """P(BINARY).isSubDomain(P(interval=1)) should return True."""
    tested = PropertyDomain(variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE)
    enclosing = PropertyDomain(interval=1)
    assert tested.isSubDomain(enclosing) is True


def test_values_against_values():
    """P(values=[1]).isSubDomain(P(values=[1, 2])) should return True."""
    tested = PropertyDomain(values=[1])
    enclosing = PropertyDomain(values=[1, 2])
    assert tested.isSubDomain(enclosing) is True


def test_binary_against_values_01():
    """P(BINARY).isSubDomain(P(values=[0, 1])) should return True."""
    tested = PropertyDomain(variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE)
    enclosing = PropertyDomain(values=[0, 1])
    assert tested.isSubDomain(enclosing) is True


def test_domain_range_interval_against_unbounded():
    """P(domainRange=[0,5], interval=1).isSubDomain(P(interval=1, no range)) should return True."""
    tested = PropertyDomain(domainRange=[0, 5], interval=1)
    enclosing = PropertyDomain(interval=1)
    assert tested.isSubDomain(enclosing) is True


def test_interval2_against_interval1():
    """P(interval=2, no range).isSubDomain(P(interval=1, no range)) should return True."""
    tested = PropertyDomain(interval=2)
    enclosing = PropertyDomain(interval=1)
    assert tested.isSubDomain(enclosing) is True


def test_unbounded_against_bounded_range():
    """P(interval=1, no range).isSubDomain(P(domainRange=[0,5], interval=1)) should return False."""
    tested = PropertyDomain(interval=1)
    enclosing = PropertyDomain(domainRange=[0, 5], interval=1)
    assert tested.isSubDomain(enclosing) is False


def test_unbounded_against_binary():
    """P(interval=1, no range).isSubDomain(P(BINARY)) should return False."""
    tested = PropertyDomain(interval=1)
    enclosing = PropertyDomain(variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE)
    assert tested.isSubDomain(enclosing) is False


def test_existing_behavior_preserved():
    """Existing subdomain behavior should still work."""
    # Identity
    d = PropertyDomain(interval=1, domainRange=[-10, 10])
    assert d.isSubDomain(d)

    # Values subset
    d1 = PropertyDomain(values=[1, 2, 3])
    d2 = PropertyDomain(values=[1, 2, 3, 4, 5])
    assert d1.isSubDomain(d2)
    assert not d2.isSubDomain(d1)

    # Binary against discrete with 0,1
    binary = PropertyDomain(variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE)
    discrete = PropertyDomain(values=[0, 1])
    assert binary.isSubDomain(discrete)
    assert discrete.isSubDomain(binary)


def test_overlaps_no_regression():
    """Ensure overlap behavior is not regressed."""
    # Discrete overlaps
    d1 = PropertyDomain(values=[1, 2, 3])
    d2 = PropertyDomain(values=[3, 4, 5])
    assert d1.overlaps(d2)
    assert d2.overlaps(d1)

    # Unbounded discrete overlaps
    unbounded = PropertyDomain(variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE, interval=1)
    assert unbounded.overlaps(d1)
    assert d1.overlaps(unbounded)

    # Binary overlaps
    binary = PropertyDomain(variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE)
    assert binary.overlaps(d1)
    assert d1.overlaps(binary)
