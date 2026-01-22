# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from typing import Any

import pytest

from orchestrator.schema.property import ConstitutiveProperty


@pytest.fixture(params=["discrete", "continuous"])
def constitutive_property_configuration_general_yaml(
    request: pytest.FixtureRequest,
) -> dict[str, Any]:
    """User this fixture for general tests of constitutive properties not requiring matching to measurement space"""

    import yaml

    confs = {
        "discrete": """
          - identifier: "AcceleratorType"
            propertyDomain:
                variableType: "CATEGORICAL_VARIABLE_TYPE"
                values: ['A100', 'T4', 1.0]
          - identifier: "BatchSize"
            propertyDomain:
              variableType: "DISCRETE_VARIABLE_TYPE"
              domainRange: [1, 500]
              interval: 1""",
        "continuous": """
          - identifier: "AcceleratorType"
            propertyDomain:
                variableType: "CATEGORICAL_VARIABLE_TYPE"
                values: ['A100', 'T4', 'A10']
          - identifier: "BatchSize"
            propertyDomain:
              variableType: "DISCRETE_VARIABLE_TYPE"
              domainRange: [1, 500]
              interval: 1
          - identifier: "MyVar"
            propertyDomain:
              variableType: "CONTINUOUS_VARIABLE_TYPE"
              domainRange: [1, 500]""",
    }

    return yaml.safe_load(confs[request.param])


@pytest.fixture
def constitutive_property_configuration_general(
    constitutive_property_configuration_general_yaml: dict[str, Any],  # noqa: ANN401
) -> list[ConstitutiveProperty]:

    return [
        ConstitutiveProperty(**p)
        for p in constitutive_property_configuration_general_yaml
    ]
