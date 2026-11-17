# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for the constraints field of DiscoverySpaceConfiguration."""

import pydantic
import pytest
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]

from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.schema.constraint import PropertyConstraint
from ado.utilities.output import pydantic_model_as_yaml

_ENTITY_SPACE = [
    {
        "identifier": "num_gpus",
        "propertyDomain": {
            "variableType": "DISCRETE_VARIABLE_TYPE",
            "values": [1, 2, 4, 8],
        },
    },
    {
        "identifier": "model",
        "propertyDomain": {
            "variableType": "CATEGORICAL_VARIABLE_TYPE",
            "values": ["llama-8b", "llama-70b"],
        },
    },
]

_LARGE_MODEL_NEEDS_MULTIPLE_GPUS = {
    "name": "large_model_needs_multiple_gpus",
    "when": {
        "left": {"propertyIdentifier": "model"},
        "operator": "eq",
        "right": {"value": "llama-70b"},
    },
    "require": {
        "left": {"propertyIdentifier": "num_gpus"},
        "operator": "ge",
        "right": {"value": 4},
    },
}


def test_configuration_constraints_default_to_empty() -> None:
    configuration = DiscoverySpaceConfiguration(entitySpace=_ENTITY_SPACE)

    assert configuration.constraints == []


def test_configuration_constraints_lifecycle() -> None:
    configuration = DiscoverySpaceConfiguration.model_validate(
        {
            "entitySpace": _ENTITY_SPACE,
            "constraints": [_LARGE_MODEL_NEEDS_MULTIPLE_GPUS],
        }
    )

    assert configuration.constraints == [
        PropertyConstraint.model_validate(_LARGE_MODEL_NEEDS_MULTIPLE_GPUS)
    ]

    from_dump = DiscoverySpaceConfiguration.model_validate(configuration.model_dump())
    assert from_dump == configuration

    from_yaml = DiscoverySpaceConfiguration.model_validate(
        yaml.load(pydantic_model_as_yaml(configuration), Loader=SafeLoader)
    )
    assert from_yaml == configuration


def test_configuration_rejects_constraint_on_unknown_property() -> None:
    unknown_property_constraint = {
        "name": "needs_learning_rate",
        "require": {
            "left": {"propertyIdentifier": "learning_rate"},
            "operator": "gt",
            "right": {"value": 0},
        },
    }

    with pytest.raises(pydantic.ValidationError, match="learning_rate"):
        DiscoverySpaceConfiguration.model_validate(
            {
                "entitySpace": _ENTITY_SPACE,
                "constraints": [unknown_property_constraint],
            }
        )


def test_configuration_rejects_duplicate_constraint_names() -> None:
    with pytest.raises(
        pydantic.ValidationError, match="large_model_needs_multiple_gpus"
    ):
        DiscoverySpaceConfiguration.model_validate(
            {
                "entitySpace": _ENTITY_SPACE,
                "constraints": [
                    _LARGE_MODEL_NEEDS_MULTIPLE_GPUS,
                    _LARGE_MODEL_NEEDS_MULTIPLE_GPUS,
                ],
            }
        )


def test_configuration_rejects_constraints_without_entity_space() -> None:
    with pytest.raises(pydantic.ValidationError, match="entitySpace"):
        DiscoverySpaceConfiguration.model_validate(
            {"constraints": [_LARGE_MODEL_NEEDS_MULTIPLE_GPUS]}
        )
