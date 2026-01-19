# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pytest

from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum


@pytest.fixture(
    params=[
        "property_domain_explicit_binary",
        "property_domain_explicit_binary_with_values",
        "property_domain_explicit_categorical_only_numbers",
        "property_domain_explicit_categorical",
        "property_domain_explicit_continuous",
        "property_domain_explicit_continuous_with_domain_range",
        "property_domain_explicit_discrete_with_domain_range_and_interval",
        "property_domain_explicit_discrete_with_interval",
        "property_domain_explicit_discrete_with_values",
        "property_domain_explicit_identifier",
        "property_domain_explicit_unknown",
        "property_domain_implicit_categorical",
        "property_domain_implicit_continuous_with_domain_range",
        "property_domain_implicit_discrete_with_domain_range_and_interval",
        "property_domain_implicit_discrete_with_interval",
        "property_domain_implicit_discrete_with_values",
        "property_domain_implicit_unknown",
    ]
)
def property_domain_all_types(request: pytest.FixtureRequest) -> PropertyDomain:
    return request.getfixturevalue(request.param)


# <----------- EXPLICIT FIXTURES ----------->
@pytest.fixture
def property_domain_explicit_binary() -> PropertyDomain:
    return PropertyDomain(variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE)


@pytest.fixture
def property_domain_explicit_binary_with_values() -> PropertyDomain:
    return PropertyDomain(
        values=[False, True], variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE
    )


@pytest.fixture
def property_domain_explicit_categorical_only_numbers() -> PropertyDomain:
    return PropertyDomain(
        values=[1, 2, 3, 4], variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
    )


@pytest.fixture
def property_domain_explicit_categorical() -> PropertyDomain:
    return PropertyDomain(
        values=[1, "B", 3, "C"], variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
    )


@pytest.fixture
def property_domain_explicit_continuous() -> PropertyDomain:
    return PropertyDomain(variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE)


@pytest.fixture
def property_domain_explicit_continuous_with_domain_range() -> PropertyDomain:
    return PropertyDomain(
        domainRange=[0, 10], variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
    )


@pytest.fixture
def property_domain_explicit_discrete_with_values() -> PropertyDomain:
    return PropertyDomain(
        values=[1, 3, 5, 6, 8], variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    )


@pytest.fixture
def property_domain_explicit_discrete_with_interval() -> PropertyDomain:
    return PropertyDomain(
        interval=1, variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    )


@pytest.fixture
def property_domain_explicit_discrete_with_domain_range_and_interval() -> (
    PropertyDomain
):
    return PropertyDomain(
        interval=2,
        domainRange=[5, 12],
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
    )


@pytest.fixture
def property_domain_explicit_identifier() -> PropertyDomain:
    import uuid

    return PropertyDomain(
        values=[str(uuid.uuid4())] * 4,
        variableType=VariableTypeEnum.IDENTIFIER_VARIABLE_TYPE,
    )


@pytest.fixture
def property_domain_explicit_unknown() -> PropertyDomain:
    return PropertyDomain(variableType=VariableTypeEnum.UNKNOWN_VARIABLE_TYPE)


# <----------- IMPLICIT FIXTURES ----------->
@pytest.fixture
def property_domain_implicit_categorical() -> PropertyDomain:
    return PropertyDomain(values=[1, "B", 3, "C"])


@pytest.fixture
def property_domain_implicit_continuous_with_domain_range() -> PropertyDomain:
    return PropertyDomain(domainRange=[0, 10])


@pytest.fixture
def property_domain_implicit_discrete_with_values() -> PropertyDomain:
    return PropertyDomain(values=[1, 3, 5, 6, 8])


@pytest.fixture
def property_domain_implicit_discrete_with_interval() -> PropertyDomain:
    return PropertyDomain(interval=1)


@pytest.fixture
def property_domain_implicit_discrete_with_domain_range_and_interval() -> (
    PropertyDomain
):
    return PropertyDomain(
        interval=2,
        domainRange=[5, 12],
    )


@pytest.fixture
def property_domain_implicit_unknown() -> PropertyDomain:
    return PropertyDomain()
