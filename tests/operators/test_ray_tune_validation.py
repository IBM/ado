# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for Ray Tune operator points_to_evaluate validation."""

import pytest
from ado_ray_tune.operator import _validate_points_to_evaluate

from ado.schema.domain import PropertyDomain
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.property import ConstitutiveProperty


def _pigeon10_entity_space() -> EntitySpaceRepresentation:
    """Entity space matching cplex_mip_pigeon10: mps_file, node_selection, variable_selection."""
    mps_file = ConstitutiveProperty(
        identifier="mps_file",
        propertyDomain=PropertyDomain(values=["pigeon-10.mps.gz"]),
    )
    node_selection = ConstitutiveProperty(
        identifier="node_selection",
        propertyDomain=PropertyDomain(values=[0, 1, 2, 3]),
    )
    variable_selection = ConstitutiveProperty(
        identifier="variable_selection",
        propertyDomain=PropertyDomain(values=[0, 1, 2, 3]),
    )
    return EntitySpaceRepresentation([mps_file, node_selection, variable_selection])


def _minimal_entity_space() -> EntitySpaceRepresentation:
    """Minimal 2-property entity space for unit tests."""
    cp1 = ConstitutiveProperty(
        identifier="a",
        propertyDomain=PropertyDomain(values=[1, 2, 3]),
    )
    cp2 = ConstitutiveProperty(
        identifier="b",
        propertyDomain=PropertyDomain(values=["x", "y"]),
    )
    return EntitySpaceRepresentation([cp1, cp2])


def test_validate_points_to_evaluate_none_passes() -> None:
    """None or empty points_to_evaluate should pass without error."""
    entity_space = _minimal_entity_space()
    _validate_points_to_evaluate(None, entity_space)
    _validate_points_to_evaluate([], entity_space)


def test_validate_points_to_evaluate_valid_point_passes() -> None:
    """Valid complete points should pass."""
    entity_space = _minimal_entity_space()
    _validate_points_to_evaluate(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
        entity_space,
    )


def test_validate_points_to_evaluate_missing_property_raises() -> None:
    """Point missing a constitutive property should raise ValueError."""
    entity_space = _minimal_entity_space()
    with pytest.raises(ValueError, match=r"missing properties.*\bb\b"):
        _validate_points_to_evaluate([{"a": 1}], entity_space)


def test_validate_points_to_evaluate_extra_property_raises() -> None:
    """Point with extra property not in space should raise ValueError."""
    entity_space = _minimal_entity_space()
    with pytest.raises(ValueError, match=r"extra properties.*\bc\b"):
        _validate_points_to_evaluate([{"a": 1, "b": "x", "c": 0}], entity_space)


def test_validate_points_to_evaluate_value_out_of_domain_raises() -> None:
    """Point with value outside constitutive property domain should raise ValueError."""
    entity_space = _minimal_entity_space()
    with pytest.raises(ValueError, match=r"invalid"):
        _validate_points_to_evaluate([{"a": 99, "b": "x"}], entity_space)


def test_validate_points_to_evaluate_non_dict_raises() -> None:
    """Point that is not a dict should raise ValueError."""
    entity_space = _minimal_entity_space()
    with pytest.raises(ValueError, match=r"must be a dict.*got list"):
        _validate_points_to_evaluate([{"a": 1, "b": "x"}, [1, 2, 3]], entity_space)


def test_validate_points_to_evaluate_pigeon10_space_bab6_points_raises() -> None:
    """Regression: bab6 points_to_evaluate (missing mps_file) on pigeon10 space raises."""
    entity_space = _pigeon10_entity_space()
    # Points from operation_lhs.yaml - designed for bab6, missing mps_file for pigeon10
    invalid_points = [
        {
            "n_threads": 1,
            "rins_frequency": 0,
            "cut_passes": 0,
            "node_selection": 1,
            "variable_selection": 0,
        }
    ]
    with pytest.raises(ValueError, match=r"missing properties.*mps_file"):
        _validate_points_to_evaluate(invalid_points, entity_space)


def test_validate_points_to_evaluate_pigeon10_space_valid_points_passes() -> None:
    """Complete pigeon10 points should pass."""
    entity_space = _pigeon10_entity_space()
    valid_points = [
        {
            "mps_file": "pigeon-10.mps.gz",
            "node_selection": 1,
            "variable_selection": 0,
        }
    ]
    _validate_points_to_evaluate(valid_points, entity_space)
