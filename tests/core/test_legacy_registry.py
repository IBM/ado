# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for the legacy migrator registry"""

from collections.abc import Callable

import pytest

from orchestrator.core.legacy.metadata import LegacyMigratorMetadata
from orchestrator.core.legacy.registry import (
    LegacyMigratorRegistry,
    legacy_migrator,
)
from orchestrator.core.resources import CoreResourceKinds


@pytest.fixture
def dummy_migrator() -> Callable[[dict], dict]:
    """Fixture providing a simple dummy migrator function"""

    def migrator(data: dict) -> dict:
        return data

    return migrator


@pytest.fixture
def create_migrator_metadata(
    dummy_migrator: Callable[[dict], dict],
) -> Callable[..., LegacyMigratorMetadata]:
    """Fixture factory for creating LegacyMigratorMetadata instances"""

    def _create_metadata(
        identifier: str = "test_migrator",
        resource_type: CoreResourceKinds = CoreResourceKinds.SAMPLESTORE,
        deprecated_field_paths: list[str] | None = None,
        deprecated_from_version: str = "1.0.0",
        removed_from_version: str = "2.0.0",
        description: str = "Test migrator",
        migrator_function: Callable[[dict], dict] | None = None,
        dependencies: list[str] | None = None,
    ) -> LegacyMigratorMetadata:
        if deprecated_field_paths is None:
            deprecated_field_paths = ["config.field1"]
        if migrator_function is None:
            migrator_function = dummy_migrator
        if dependencies is None:
            dependencies = []

        return LegacyMigratorMetadata(
            identifier=identifier,
            resource_type=resource_type,
            deprecated_field_paths=deprecated_field_paths,
            deprecated_from_version=deprecated_from_version,
            removed_from_version=removed_from_version,
            description=description,
            migrator_function=migrator_function,
            dependencies=dependencies,
        )

    return _create_metadata


class TestLegacyMigratorMetadata:
    """Test the LegacyMigratorMetadata model"""

    def test_create_metadata(
        self,
        create_migrator_metadata: Callable[..., LegacyMigratorMetadata],
        dummy_migrator: Callable[[dict], dict],
    ) -> None:
        """Test creating migrator metadata"""
        metadata = create_migrator_metadata(
            identifier="test_migrator",
            deprecated_field_paths=["config.field1", "config.field2"],
        )

        assert metadata.identifier == "test_migrator"
        assert metadata.resource_type == CoreResourceKinds.SAMPLESTORE
        assert metadata.deprecated_field_paths == [
            "config.field1",
            "config.field2",
        ]
        assert metadata.deprecated_from_version == "1.0.0"
        assert metadata.removed_from_version == "2.0.0"
        assert metadata.description == "Test migrator"
        assert metadata.migrator_function == dummy_migrator

    def test_metadata_serialization(
        self, create_migrator_metadata: Callable[..., LegacyMigratorMetadata]
    ) -> None:
        """Test that migrator function is excluded from serialization"""
        metadata = create_migrator_metadata()

        # Serialize to dict
        data = metadata.model_dump()

        # migrator_function should be excluded
        assert "migrator_function" not in data
        assert "identifier" in data
        assert "resource_type" in data


class TestLegacyMigratorRegistry:
    """Test the LegacyMigratorRegistry class"""

    def test_register_migrator(
        self,
        isolated_legacy_migrator_registry: None,
        create_migrator_metadata: Callable[..., LegacyMigratorMetadata],
    ) -> None:
        """Test registering a migrator"""
        metadata = create_migrator_metadata()

        LegacyMigratorRegistry.register(metadata)

        assert len(LegacyMigratorRegistry._migrators) == 1
        assert "test_migrator" in LegacyMigratorRegistry._migrators

    def test_get_migrator(
        self,
        isolated_legacy_migrator_registry: None,
        create_migrator_metadata: Callable[..., LegacyMigratorMetadata],
    ) -> None:
        """Test retrieving a migrator by identifier"""
        metadata = create_migrator_metadata()

        LegacyMigratorRegistry.register(metadata)

        retrieved = LegacyMigratorRegistry.get_migrator("test_migrator")
        assert retrieved is not None
        assert retrieved.identifier == "test_migrator"

    def test_get_nonexistent_migrator(
        self, isolated_legacy_migrator_registry: None
    ) -> None:
        """Test retrieving a migrator that doesn't exist"""
        retrieved = LegacyMigratorRegistry.get_migrator("nonexistent")
        assert retrieved is None

    def test_get_migrators_for_resource(
        self,
        isolated_legacy_migrator_registry: None,
        create_migrator_metadata: Callable[..., LegacyMigratorMetadata],
    ) -> None:
        """Test retrieving validators for a specific resource type"""
        # Register validators for different resource types
        metadata1 = create_migrator_metadata(
            identifier="samplestore_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            description="Sample store migrator",
        )

        metadata2 = create_migrator_metadata(
            identifier="operation_migrator",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_field_paths=["config.field2"],
            description="Operation migrator",
        )

        LegacyMigratorRegistry.register(metadata1)
        LegacyMigratorRegistry.register(metadata2)

        # Get validators for SAMPLESTORE
        samplestore_migrators = LegacyMigratorRegistry.get_migrators_for_resource(
            CoreResourceKinds.SAMPLESTORE
        )
        assert len(samplestore_migrators) == 1
        assert samplestore_migrators[0].identifier == "samplestore_migrator"

        # Get validators for OPERATION
        operation_migrators = LegacyMigratorRegistry.get_migrators_for_resource(
            CoreResourceKinds.OPERATION
        )
        assert len(operation_migrators) == 1
        assert operation_migrators[0].identifier == "operation_migrator"

    def test_find_migrators_for_deprecated_field_paths(
        self,
        isolated_legacy_migrator_registry: None,
        create_migrator_metadata: Callable[..., LegacyMigratorMetadata],
    ) -> None:
        """Test finding validators that handle specific field paths"""
        # Register validators with different field paths
        metadata1 = create_migrator_metadata(
            identifier="migrator1",
            deprecated_field_paths=["config.field1", "config.field2"],
            description="Validator 1",
        )

        metadata2 = create_migrator_metadata(
            identifier="migrator2",
            deprecated_field_paths=["config.specification.field3"],
            description="Validator 2",
        )

        metadata3 = create_migrator_metadata(
            identifier="migrator3",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.properties"],
            description="Validator 3",
        )

        LegacyMigratorRegistry.register(metadata1)
        LegacyMigratorRegistry.register(metadata2)
        LegacyMigratorRegistry.register(metadata3)

        # Find validators for single full path
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.field1"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "migrator1"

        # Find validators for nested path
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.specification.field3"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "migrator2"

        # Find validators for multiple paths
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE,
            {"config.field1", "config.specification.field3"},
        )
        assert len(validators) == 2
        migrator_ids = {v.identifier for v in validators}
        assert migrator_ids == {"migrator1", "migrator2"}

        # Find validators for non-existent path
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.nonexistent"}
        )
        assert len(validators) == 0

        # Verify it doesn't match on leaf names alone (more specific than find_migrators_for_fields)
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"field1"}  # Just leaf name, not full path
        )
        assert len(validators) == 0

        # Verify resource type filtering works
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.DISCOVERYSPACE, {"config.properties"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "migrator3"

    def test_list_all(
        self,
        isolated_legacy_migrator_registry: None,
        create_migrator_metadata: Callable[..., LegacyMigratorMetadata],
    ) -> None:
        """Test listing all validators"""
        metadata1 = create_migrator_metadata(
            identifier="migrator1",
            description="Validator 1",
        )

        metadata2 = create_migrator_metadata(
            identifier="migrator2",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_field_paths=["config.field2"],
            description="Validator 2",
        )

        LegacyMigratorRegistry.register(metadata1)
        LegacyMigratorRegistry.register(metadata2)

        all_migrators = LegacyMigratorRegistry.list_all()
        assert len(all_migrators) == 2

    def test_field_path_matching_with_real_migrators(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Integration test: verify field path matching works with real validators"""

        # Test 1: discoveryspace properties field should match the properties_field_removal validator
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.DISCOVERYSPACE, {"config.properties"}
        )
        assert len(validators) >= 1
        migrator_ids = {v.identifier for v in validators}
        assert "discoveryspace_properties_field_removal" in migrator_ids

        # Test 2: operation actuators field should match the actuators_field_removal validator
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.OPERATION, {"config.actuators"}
        )
        assert len(validators) >= 1
        migrator_ids = {v.identifier for v in validators}
        assert "operation_actuators_field_removal" in migrator_ids

        # Test 3: operation parameters.mode should match randomwalk validator
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.OPERATION, {"config.parameters.mode"}
        )
        assert len(validators) >= 1
        migrator_ids = {v.identifier for v in validators}
        assert "randomwalk_mode_to_sampler_config" in migrator_ids

        # Test 4: samplestore config.specification.module.moduleType should match the module_type validator
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.specification.module.moduleType"}
        )
        assert len(validators) >= 1
        migrator_ids = {v.identifier for v in validators}
        assert "samplestore_module_type_entitysource_to_samplestore" in migrator_ids

        # Test 5: samplestore kind field should match the kind validator
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"kind"}
        )
        assert len(validators) >= 1
        migrator_ids = {v.identifier for v in validators}
        assert "samplestore_kind_entitysource_to_samplestore" in migrator_ids

        # Test 6: Multiple paths should return multiple validators
        validators = LegacyMigratorRegistry.find_migrators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE,
            {
                "config.specification.module.moduleType",
                "config.specification.module.moduleClass",
                "config.specification.module.moduleName",
            },
        )
        assert len(validators) >= 3
        migrator_ids = {v.identifier for v in validators}
        assert "samplestore_module_type_entitysource_to_samplestore" in migrator_ids
        assert "samplestore_module_class_entitysource_to_samplestore" in migrator_ids
        assert "samplestore_module_name_entitysource_to_samplestore" in migrator_ids


class TestLegacyMigratorDecorator:
    """Test the @legacy_migrator decorator"""

    def test_decorator_registers_migrator(
        self, isolated_legacy_migrator_registry: None
    ) -> None:
        """Test that the decorator registers the migrator"""

        @legacy_migrator(
            identifier="test_decorator_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test decorator migrator",
        )
        def my_migrator(data: dict) -> dict:
            return data

        # Check that migrator was registered
        assert len(LegacyMigratorRegistry._migrators) == 1
        assert "test_decorator_migrator" in LegacyMigratorRegistry._migrators

        # Check that the function still works
        test_data = {"key": "value"}
        result = my_migrator(test_data)
        assert result == test_data

    def test_decorator_preserves_function_metadata(
        self, isolated_legacy_migrator_registry: None
    ) -> None:
        """Test that the decorator preserves function metadata"""

        @legacy_migrator(
            identifier="test_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test migrator",
        )
        def my_migrator(data: dict) -> dict:
            """My migrator docstring"""
            return data

        # Check that function name and docstring are preserved
        assert my_migrator.__name__ == "my_migrator"
        assert my_migrator.__doc__ == "My migrator docstring"

    def test_migrator_function_execution(self) -> None:
        """Test that the migrator function executes correctly"""

        @legacy_migrator(
            identifier="transform_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Transform migrator",
        )
        def transform_migrator(data: dict) -> dict:
            if "old_field" in data:
                data["new_field"] = data.pop("old_field")
            return data

        # Test the migrator function
        test_data = {"old_field": "value"}
        result = transform_migrator(test_data)
        assert "old_field" not in result
        assert result["new_field"] == "value"

        # Verify it was registered correctly
        metadata = LegacyMigratorRegistry.get_migrator("transform_migrator")
        assert metadata is not None
        # The migrator function should be callable and work correctly
        test_data2 = {"old_field": "another_value"}
        result2 = metadata.migrator_function(test_data2)
        assert "old_field" not in result2
        assert result2["new_field"] == "another_value"

    # Made with Bob
