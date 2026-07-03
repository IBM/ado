# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import math

import pydantic
import pytest

from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.vector_domain import VectorPropertyDomain


@pytest.fixture
def simple_element_domain():
    # Discrete domain: {1, 2, 3}
    return PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        values=[1, 2, 3],
    )


def test_vector_property_domain_valid_vector(simple_element_domain):
    vpd = VectorPropertyDomain(element_domain=simple_element_domain, number_elements=2)
    assert vpd.valueInDomain([1, 2])
    assert vpd.valueInDomain([2, 3])
    assert not vpd.valueInDomain([1, 999])  # 999 not in element domain
    assert not vpd.valueInDomain([1])  # Too short
    assert not vpd.valueInDomain([1, 2, 3])  # Too long


def test_vector_property_domain_domain_values(simple_element_domain):
    vpd = VectorPropertyDomain(element_domain=simple_element_domain, number_elements=2)
    values = vpd.domain_values
    # Should be the cartesian product
    expected = [(a, b) for a in [1, 2, 3] for b in [1, 2, 3]]
    assert set(values) == set(expected)
    assert len(values) == 9  # 3^2


def test_vector_property_domain_size(simple_element_domain):
    vpd = VectorPropertyDomain(element_domain=simple_element_domain, number_elements=3)
    assert vpd.size == 27
    # Inf if element_domain not countable
    # Make continuous domain (should not allow domain_values)
    from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum

    cd = PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE, domainRange=[0, 1]
    )
    vpd_cont = VectorPropertyDomain(element_domain=cd, number_elements=2)
    assert math.isinf(vpd_cont.size)
    with pytest.raises(Exception, match="element_domain must be discrete"):
        _ = vpd_cont.domain_values


def test_vector_property_domain_isSubDomain(simple_element_domain):
    eldom_small = PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE, values=[1]
    )
    eldom_big = PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE, values=[1, 2, 3]
    )
    vpd_small = VectorPropertyDomain(element_domain=eldom_small, number_elements=2)
    vpd_big = VectorPropertyDomain(element_domain=eldom_big, number_elements=3)
    # Check: fewer dims
    assert vpd_small.isSubDomain(vpd_big)
    # Reverse: should fail (more dims)
    assert not vpd_big.isSubDomain(vpd_small)
    # Same number dims but element subdomain wrong
    vpd2 = VectorPropertyDomain(element_domain=eldom_big, number_elements=2)
    assert not vpd2.isSubDomain(vpd_small)


def test_vector_property_domain_eq(simple_element_domain):
    vpd1 = VectorPropertyDomain(element_domain=simple_element_domain, number_elements=2)
    vpd2 = VectorPropertyDomain(element_domain=simple_element_domain, number_elements=2)
    assert vpd1 == vpd2
    vpd3 = VectorPropertyDomain(element_domain=simple_element_domain, number_elements=3)
    assert vpd1 != vpd3


def test_vector_property_domain_variableType_guard(simple_element_domain):
    # If someone tries to construct it with wrong variableType, should raise error

    from orchestrator.schema.domain import VariableTypeEnum

    with pytest.raises(pydantic.ValidationError, match="VariableType must be VECTOR"):
        VectorPropertyDomain(
            element_domain=simple_element_domain,
            number_elements=2,
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        )
