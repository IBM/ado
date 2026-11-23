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
from ado.schema.property import ConstitutiveProperty
from ado.utilities.output import pydantic_model_as_yaml


def test_configuration_constraints_lifecycle(
    constitutive_property_configuration_general: list[ConstitutiveProperty],
    t4_batch_size_constraint: PropertyConstraint,
) -> None:
    """Test constraints survive model dump and YAML load."""
    configuration = DiscoverySpaceConfiguration(
        entitySpace=constitutive_property_configuration_general,
        constraints=[t4_batch_size_constraint],
    )

    from_dump = DiscoverySpaceConfiguration.model_validate(configuration.model_dump())
    assert from_dump == configuration

    from_yaml = DiscoverySpaceConfiguration.model_validate(
        yaml.load(pydantic_model_as_yaml(configuration), Loader=SafeLoader)
    )
    assert from_yaml == configuration


def test_configuration_rejects_constraint_on_property_not_in_entity_space(
    constitutive_property_configuration_general: list[ConstitutiveProperty],
    t4_batch_size_constraint: PropertyConstraint,
) -> None:
    """Test a constraint on a property absent from the entity space is rejected."""
    properties_without_batch_size = [
        p
        for p in constitutive_property_configuration_general
        if p.identifier != "BatchSize"
    ]

    with pytest.raises(pydantic.ValidationError, match="BatchSize"):
        DiscoverySpaceConfiguration(
            entitySpace=properties_without_batch_size,
            constraints=[t4_batch_size_constraint],
        )


def test_configuration_rejects_constraints_without_entity_space(
    t4_batch_size_constraint: PropertyConstraint,
) -> None:
    """Test constraints without an entity space are rejected."""
    with pytest.raises(pydantic.ValidationError, match="entitySpace"):
        DiscoverySpaceConfiguration(constraints=[t4_batch_size_constraint])
