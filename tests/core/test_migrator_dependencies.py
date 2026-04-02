# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for migrator dependency resolution and ordering"""

import pytest

from orchestrator.core.legacy.metadata import LegacyMigratorMetadata
from orchestrator.core.legacy.registry import LegacyMigratorRegistry
from orchestrator.core.resources import CoreResourceKinds


def test_resolve_dependencies_no_dependencies(
    isolated_legacy_migrator_registry: None,
) -> None:
    """Test resolving validators with no dependencies"""

    def migrator_a(data: dict) -> dict:
        return data

    def migrator_b(data: dict) -> dict:
        return data

    # Register validators without dependencies
    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            migrator_function=migrator_a,
            dependencies=[],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            migrator_function=migrator_b,
            dependencies=[],
        )
    )

    # Resolve dependencies
    ordered, missing = LegacyMigratorRegistry.resolve_dependencies(
        ["migrator_a", "migrator_b"]
    )

    # Should return both validators in alphabetical order (no dependencies)
    assert len(ordered) == 2
    assert "migrator_a" in ordered
    assert "migrator_b" in ordered
    assert len(missing) == 0


def test_resolve_dependencies_simple_chain() -> None:
    """Test resolving validators with simple dependency chain"""

    def migrator_a(data: dict) -> dict:
        return data

    def migrator_b(data: dict) -> dict:
        return data

    def migrator_c(data: dict) -> dict:
        return data

    # Register validators: C depends on B, B depends on A
    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            migrator_function=migrator_a,
            dependencies=[],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            migrator_function=migrator_b,
            dependencies=["migrator_a"],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_c",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_c"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator C",
            migrator_function=migrator_c,
            dependencies=["migrator_b"],
        )
    )

    # Resolve dependencies - only request C
    ordered, missing = LegacyMigratorRegistry.resolve_dependencies(["migrator_c"])

    # Should return all three in correct order: A, B, C
    assert ordered == ["migrator_a", "migrator_b", "migrator_c"]
    assert len(missing) == 0


def test_resolve_dependencies_diamond(
    isolated_legacy_migrator_registry: None,
) -> None:
    """Test resolving validators with diamond dependency pattern"""

    def migrator_a(data: dict) -> dict:
        return data

    def migrator_b(data: dict) -> dict:
        return data

    def migrator_c(data: dict) -> dict:
        return data

    def migrator_d(data: dict) -> dict:
        return data

    # Register validators: D depends on B and C, both B and C depend on A
    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            migrator_function=migrator_a,
            dependencies=[],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            migrator_function=migrator_b,
            dependencies=["migrator_a"],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_c",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_c"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator C",
            migrator_function=migrator_c,
            dependencies=["migrator_a"],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_d",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_d"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator D",
            migrator_function=migrator_d,
            dependencies=["migrator_b", "migrator_c"],
        )
    )

    # Resolve dependencies
    ordered, missing = LegacyMigratorRegistry.resolve_dependencies(["migrator_d"])

    # Should return all four: A first, then B and C (in some order), then D
    assert len(ordered) == 4
    assert ordered[0] == "migrator_a"  # A must be first
    assert ordered[3] == "migrator_d"  # D must be last
    assert "migrator_b" in ordered[1:3]  # B and C in middle
    assert "migrator_c" in ordered[1:3]
    assert len(missing) == 0


def test_resolve_dependencies_circular(
    isolated_legacy_migrator_registry: None,
) -> None:
    """Test that circular dependencies are detected"""

    def migrator_a(data: dict) -> dict:
        return data

    def migrator_b(data: dict) -> dict:
        return data

    # Register validators with circular dependency: A depends on B, B depends on A
    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            migrator_function=migrator_a,
            dependencies=["migrator_b"],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            migrator_function=migrator_b,
            dependencies=["migrator_a"],
        )
    )

    # Should raise ValueError for circular dependency
    with pytest.raises(ValueError, match="Circular dependency detected"):
        LegacyMigratorRegistry.resolve_dependencies(["migrator_a", "migrator_b"])


def test_resolve_dependencies_missing(
    isolated_legacy_migrator_registry: None,
) -> None:
    """Test handling of missing dependencies"""

    def migrator_a(data: dict) -> dict:
        return data

    # Register migrator with non-existent dependency
    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            migrator_function=migrator_a,
            dependencies=["nonexistent_migrator"],
        )
    )

    # Resolve dependencies
    ordered, missing = LegacyMigratorRegistry.resolve_dependencies(["migrator_a"])

    # Should return migrator_a and report missing dependency
    assert ordered == ["migrator_a"]
    assert "nonexistent_migrator" in missing


def test_resolve_dependencies_multiple_roots(
    isolated_legacy_migrator_registry: None,
) -> None:
    """Test resolving validators with multiple independent roots"""

    def migrator_a(data: dict) -> dict:
        return data

    def migrator_b(data: dict) -> dict:
        return data

    def migrator_c(data: dict) -> dict:
        return data

    def migrator_d(data: dict) -> dict:
        return data

    # Register validators: C depends on A, D depends on B
    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            migrator_function=migrator_a,
            dependencies=[],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            migrator_function=migrator_b,
            dependencies=[],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_c",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_c"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator C",
            migrator_function=migrator_c,
            dependencies=["migrator_a"],
        )
    )

    LegacyMigratorRegistry.register(
        LegacyMigratorMetadata(
            identifier="migrator_d",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_d"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator D",
            migrator_function=migrator_d,
            dependencies=["migrator_b"],
        )
    )

    # Resolve dependencies
    ordered, missing = LegacyMigratorRegistry.resolve_dependencies(
        ["migrator_c", "migrator_d"]
    )

    # Should return all four validators with correct ordering
    assert len(ordered) == 4
    # A must come before C
    assert ordered.index("migrator_a") < ordered.index("migrator_c")
    # B must come before D
    assert ordered.index("migrator_b") < ordered.index("migrator_d")
    assert len(missing) == 0


# Made with Bob
