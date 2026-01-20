# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pytest

from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
)


@pytest.fixture
def target_property_list() -> list[str]:
    return ["pka", "wall_time", "ld50"]


@pytest.fixture
def abstract_properties(
    target_property_list: list[str],
) -> list[AbstractPropertyDescriptor]:
    return [AbstractPropertyDescriptor(identifier=t) for t in target_property_list]


@pytest.fixture
def constitutive_property_list() -> list[str]:
    return ["number-carbons", "is-ionic", "total-electrons"]


@pytest.fixture
def constitutive_properties(
    constitutive_property_list: list[str],
) -> list[ConstitutiveProperty]:
    return [ConstitutiveProperty(identifier=t) for t in constitutive_property_list]


@pytest.fixture(scope="module")
def requiredProperties() -> list[ConstitutiveProperty]:

    return [
        ConstitutiveProperty(
            identifier="test_req1",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                values=["X", "Y", "W"],
            ),
        ),
        ConstitutiveProperty(
            identifier="test_req2",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
                domainRange=[0, 20],
                interval=1,
            ),
        ),
        ConstitutiveProperty(
            identifier="test_req3",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
                values=[2, 4, 5, 6, 8],
            ),
        ),
    ]
