# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for validator dependency resolution and ordering"""

from collections.abc import Generator

import pytest

from orchestrator.core.legacy.metadata import LegacyValidatorMetadata
from orchestrator.core.legacy.registry import LegacyValidatorRegistry
from orchestrator.core.resources import CoreResourceKinds


@pytest.fixture(autouse=True)
def clear_registry() -> Generator[None, None, None]:
    """Clear the registry before and after each test"""
    LegacyValidatorRegistry._validators.clear()
    yield
    LegacyValidatorRegistry._validators.clear()


def test_resolve_dependencies_no_dependencies() -> None:
    """Test resolving validators with no dependencies"""

    def validator_a(data: dict) -> dict:
        return data

    def validator_b(data: dict) -> dict:
        return data

    # Register validators without dependencies
    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            validator_function=validator_a,
            dependencies=[],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            validator_function=validator_b,
            dependencies=[],
        )
    )

    # Resolve dependencies
    ordered, missing = LegacyValidatorRegistry.resolve_dependencies(
        ["validator_a", "validator_b"]
    )

    # Should return both validators in alphabetical order (no dependencies)
    assert len(ordered) == 2
    assert "validator_a" in ordered
    assert "validator_b" in ordered
    assert len(missing) == 0


def test_resolve_dependencies_simple_chain() -> None:
    """Test resolving validators with simple dependency chain"""

    def validator_a(data: dict) -> dict:
        return data

    def validator_b(data: dict) -> dict:
        return data

    def validator_c(data: dict) -> dict:
        return data

    # Register validators: C depends on B, B depends on A
    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            validator_function=validator_a,
            dependencies=[],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            validator_function=validator_b,
            dependencies=["validator_a"],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_c",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_c"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator C",
            validator_function=validator_c,
            dependencies=["validator_b"],
        )
    )

    # Resolve dependencies - only request C
    ordered, missing = LegacyValidatorRegistry.resolve_dependencies(["validator_c"])

    # Should return all three in correct order: A, B, C
    assert ordered == ["validator_a", "validator_b", "validator_c"]
    assert len(missing) == 0


def test_resolve_dependencies_diamond() -> None:
    """Test resolving validators with diamond dependency pattern"""

    def validator_a(data: dict) -> dict:
        return data

    def validator_b(data: dict) -> dict:
        return data

    def validator_c(data: dict) -> dict:
        return data

    def validator_d(data: dict) -> dict:
        return data

    # Register validators: D depends on B and C, both B and C depend on A
    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            validator_function=validator_a,
            dependencies=[],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            validator_function=validator_b,
            dependencies=["validator_a"],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_c",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_c"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator C",
            validator_function=validator_c,
            dependencies=["validator_a"],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_d",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_d"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator D",
            validator_function=validator_d,
            dependencies=["validator_b", "validator_c"],
        )
    )

    # Resolve dependencies
    ordered, missing = LegacyValidatorRegistry.resolve_dependencies(["validator_d"])

    # Should return all four: A first, then B and C (in some order), then D
    assert len(ordered) == 4
    assert ordered[0] == "validator_a"  # A must be first
    assert ordered[3] == "validator_d"  # D must be last
    assert "validator_b" in ordered[1:3]  # B and C in middle
    assert "validator_c" in ordered[1:3]
    assert len(missing) == 0


def test_resolve_dependencies_circular() -> None:
    """Test that circular dependencies are detected"""

    def validator_a(data: dict) -> dict:
        return data

    def validator_b(data: dict) -> dict:
        return data

    # Register validators with circular dependency: A depends on B, B depends on A
    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            validator_function=validator_a,
            dependencies=["validator_b"],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            validator_function=validator_b,
            dependencies=["validator_a"],
        )
    )

    # Should raise ValueError for circular dependency
    with pytest.raises(ValueError, match="Circular dependency detected"):
        LegacyValidatorRegistry.resolve_dependencies(["validator_a", "validator_b"])


def test_resolve_dependencies_missing() -> None:
    """Test handling of missing dependencies"""

    def validator_a(data: dict) -> dict:
        return data

    # Register validator with non-existent dependency
    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            validator_function=validator_a,
            dependencies=["nonexistent_validator"],
        )
    )

    # Resolve dependencies
    ordered, missing = LegacyValidatorRegistry.resolve_dependencies(["validator_a"])

    # Should return validator_a and report missing dependency
    assert ordered == ["validator_a"]
    assert "nonexistent_validator" in missing


def test_resolve_dependencies_multiple_roots() -> None:
    """Test resolving validators with multiple independent roots"""

    def validator_a(data: dict) -> dict:
        return data

    def validator_b(data: dict) -> dict:
        return data

    def validator_c(data: dict) -> dict:
        return data

    def validator_d(data: dict) -> dict:
        return data

    # Register validators: C depends on A, D depends on B
    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_a",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_a"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator A",
            validator_function=validator_a,
            dependencies=[],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_b",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_b"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator B",
            validator_function=validator_b,
            dependencies=[],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_c",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_c"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator C",
            validator_function=validator_c,
            dependencies=["validator_a"],
        )
    )

    LegacyValidatorRegistry.register(
        LegacyValidatorMetadata(
            identifier="validator_d",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.field_d"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator D",
            validator_function=validator_d,
            dependencies=["validator_b"],
        )
    )

    # Resolve dependencies
    ordered, missing = LegacyValidatorRegistry.resolve_dependencies(
        ["validator_c", "validator_d"]
    )

    # Should return all four validators with correct ordering
    assert len(ordered) == 4
    # A must come before C
    assert ordered.index("validator_a") < ordered.index("validator_c")
    # B must come before D
    assert ordered.index("validator_b") < ordered.index("validator_d")
    assert len(missing) == 0


# Made with Bob
