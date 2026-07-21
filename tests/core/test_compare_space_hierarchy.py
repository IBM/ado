# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for DiscoverySpaceConfiguration.compare_space_hierarchy()

Covers all six SpaceHierarchy return values:
  EQUAL_SPACE, SUB_SPACE, SUPER_SPACE, OVERLAPPING, DISJOINT, UNDEFINED
"""

from collections.abc import Callable

from ado.core.discoveryspace.config import DiscoverySpaceConfiguration, SpaceHierarchy
from ado.schema.domain import PropertyDomain


def test_equal_space(
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Identical entity spaces → EQUAL_SPACE."""
    space = make_continuous_space_configuration([0, 10])
    assert space.compare_space_hierarchy(space) == SpaceHierarchy.EQUAL_SPACE


def test_sub_space(
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Strictly smaller space → SUB_SPACE."""
    large = make_continuous_space_configuration([0, 10])
    small = make_continuous_space_configuration([2, 8])
    assert small.compare_space_hierarchy(large) == SpaceHierarchy.SUB_SPACE
    assert large.compare_space_hierarchy(small) == SpaceHierarchy.SUPER_SPACE


def test_super_space(
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Strictly larger space → SUPER_SPACE."""
    large = make_continuous_space_configuration([0, 20])
    small = make_continuous_space_configuration([5, 15])
    assert large.compare_space_hierarchy(small) == SpaceHierarchy.SUPER_SPACE
    assert small.compare_space_hierarchy(large) == SpaceHierarchy.SUB_SPACE


def test_overlapping_space(
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Domains that intersect but neither contains the other → OVERLAPPING."""
    space_a = make_continuous_space_configuration([0, 10])
    space_b = make_continuous_space_configuration([5, 15])
    assert space_a.compare_space_hierarchy(space_b) == SpaceHierarchy.OVERLAPPING
    assert space_b.compare_space_hierarchy(space_a) == SpaceHierarchy.OVERLAPPING


def test_disjoint_space(
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Domains that share no points → DISJOINT."""
    space_a = make_continuous_space_configuration([0, 5])
    space_b = make_continuous_space_configuration([10, 20])
    assert space_a.compare_space_hierarchy(space_b) == SpaceHierarchy.DISJOINT
    assert space_b.compare_space_hierarchy(space_a) == SpaceHierarchy.DISJOINT


def test_disjoint_space_no_shared_properties(
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Spaces with completely different property identifiers → DISJOINT."""
    space_a = make_continuous_space_configuration([0, 10], identifier="x")
    space_b = make_continuous_space_configuration([0, 10], identifier="y")
    assert space_a.compare_space_hierarchy(space_b) == SpaceHierarchy.DISJOINT
    assert space_b.compare_space_hierarchy(space_a) == SpaceHierarchy.DISJOINT


def test_undefined_when_entity_space_empty(
    space_configuration_none_entity_space: DiscoverySpaceConfiguration,
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Empty entity spaces → UNDEFINED (exception path)."""
    space = make_continuous_space_configuration([0, 10])
    assert (
        space_configuration_none_entity_space.compare_space_hierarchy(space)
        == SpaceHierarchy.UNDEFINED
    )
    assert (
        space.compare_space_hierarchy(space_configuration_none_entity_space)
        == SpaceHierarchy.UNDEFINED
    )


def test_overlapping_multiple_properties(
    make_multi_prop_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Overlapping result with multiple properties (all shared props must overlap)."""
    space_a = make_multi_prop_space_configuration(
        [
            ("x", PropertyDomain(domainRange=[0, 10])),
            ("y", PropertyDomain(values=["A", "B"])),
        ]
    )
    space_b = make_multi_prop_space_configuration(
        [
            ("x", PropertyDomain(domainRange=[5, 15])),
            ("y", PropertyDomain(values=["B", "C"])),
        ]
    )
    assert space_a.compare_space_hierarchy(space_b) == SpaceHierarchy.OVERLAPPING


def test_disjoint_when_one_shared_property_is_disjoint(
    make_multi_prop_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """DISJOINT when at least one shared property domain is disjoint."""
    space_a = make_multi_prop_space_configuration(
        [
            ("x", PropertyDomain(domainRange=[0, 5])),
            ("y", PropertyDomain(values=["A", "B"])),
        ]
    )
    space_b = make_multi_prop_space_configuration(
        [
            ("x", PropertyDomain(domainRange=[10, 20])),
            ("y", PropertyDomain(values=["B", "C"])),
        ]
    )
    assert space_a.compare_space_hierarchy(space_b) == SpaceHierarchy.DISJOINT


def test_overlapping_discrete_domains(
    make_discrete_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Overlapping with discrete domains sharing a value."""
    space_a = make_discrete_space_configuration([1, 2, 3])
    space_b = make_discrete_space_configuration([3, 4, 5])
    assert space_a.compare_space_hierarchy(space_b) == SpaceHierarchy.OVERLAPPING


def test_disjoint_discrete_domains(
    make_discrete_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """DISJOINT with non-intersecting discrete domains."""
    space_a = make_discrete_space_configuration([1, 2])
    space_b = make_discrete_space_configuration([3, 4])
    assert space_a.compare_space_hierarchy(space_b) == SpaceHierarchy.DISJOINT


def test_sub_space_with_extra_property(
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
    make_multi_prop_space_configuration: Callable[..., DiscoverySpaceConfiguration],
) -> None:
    """Space with fewer properties than reference is a sub-space when contained."""
    reference = make_multi_prop_space_configuration(
        [
            ("x", PropertyDomain(domainRange=[0, 10])),
            ("y", PropertyDomain(values=["A", "B"])),
        ]
    )
    sub = make_continuous_space_configuration([2, 8])
    assert sub.compare_space_hierarchy(reference) == SpaceHierarchy.SUB_SPACE
