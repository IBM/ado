# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for the legacy validator registry"""

from collections.abc import Callable

import pytest

from orchestrator.core.legacy.metadata import LegacyValidatorMetadata
from orchestrator.core.legacy.registry import (
    LegacyValidatorRegistry,
    legacy_validator,
)
from orchestrator.core.resources import CoreResourceKinds


@pytest.fixture
def dummy_validator() -> Callable[[dict], dict]:
    """Fixture providing a simple dummy validator function"""

    def validator(data: dict) -> dict:
        return data

    return validator


@pytest.fixture
def create_validator_metadata(
    dummy_validator: Callable[[dict], dict],
) -> Callable[..., LegacyValidatorMetadata]:
    """Fixture factory for creating LegacyValidatorMetadata instances"""

    def _create_metadata(
        identifier: str = "test_validator",
        resource_type: CoreResourceKinds = CoreResourceKinds.SAMPLESTORE,
        deprecated_field_paths: list[str] | None = None,
        deprecated_from_version: str = "1.0.0",
        removed_from_version: str = "2.0.0",
        description: str = "Test validator",
        validator_function: Callable[[dict], dict] | None = None,
        dependencies: list[str] | None = None,
    ) -> LegacyValidatorMetadata:
        if deprecated_field_paths is None:
            deprecated_field_paths = ["config.field1"]
        if validator_function is None:
            validator_function = dummy_validator
        if dependencies is None:
            dependencies = []

        return LegacyValidatorMetadata(
            identifier=identifier,
            resource_type=resource_type,
            deprecated_field_paths=deprecated_field_paths,
            deprecated_from_version=deprecated_from_version,
            removed_from_version=removed_from_version,
            description=description,
            validator_function=validator_function,
            dependencies=dependencies,
        )

    return _create_metadata


class TestLegacyValidatorMetadata:
    """Test the LegacyValidatorMetadata model"""

    def test_create_metadata(
        self,
        create_validator_metadata: Callable[..., LegacyValidatorMetadata],
        dummy_validator: Callable[[dict], dict],
    ) -> None:
        """Test creating validator metadata"""
        metadata = create_validator_metadata(
            identifier="test_validator",
            deprecated_field_paths=["config.field1", "config.field2"],
        )

        assert metadata.identifier == "test_validator"
        assert metadata.resource_type == CoreResourceKinds.SAMPLESTORE
        assert metadata.deprecated_field_paths == [
            "config.field1",
            "config.field2",
        ]
        assert metadata.deprecated_from_version == "1.0.0"
        assert metadata.removed_from_version == "2.0.0"
        assert metadata.description == "Test validator"
        assert metadata.validator_function == dummy_validator

    def test_metadata_serialization(
        self, create_validator_metadata: Callable[..., LegacyValidatorMetadata]
    ) -> None:
        """Test that validator function is excluded from serialization"""
        metadata = create_validator_metadata()

        # Serialize to dict
        data = metadata.model_dump()

        # validator_function should be excluded
        assert "validator_function" not in data
        assert "identifier" in data
        assert "resource_type" in data


class TestLegacyValidatorRegistry:
    """Test the LegacyValidatorRegistry class"""

    def test_register_validator(
        self,
        isolated_legacy_validator_registry: None,
        create_validator_metadata: Callable[..., LegacyValidatorMetadata],
    ) -> None:
        """Test registering a validator"""
        metadata = create_validator_metadata()

        LegacyValidatorRegistry.register(metadata)

        assert len(LegacyValidatorRegistry._validators) == 1
        assert "test_validator" in LegacyValidatorRegistry._validators

    def test_get_validator(
        self,
        isolated_legacy_validator_registry: None,
        create_validator_metadata: Callable[..., LegacyValidatorMetadata],
    ) -> None:
        """Test retrieving a validator by identifier"""
        metadata = create_validator_metadata()

        LegacyValidatorRegistry.register(metadata)

        retrieved = LegacyValidatorRegistry.get_validator("test_validator")
        assert retrieved is not None
        assert retrieved.identifier == "test_validator"

    def test_get_nonexistent_validator(
        self, isolated_legacy_validator_registry: None
    ) -> None:
        """Test retrieving a validator that doesn't exist"""
        retrieved = LegacyValidatorRegistry.get_validator("nonexistent")
        assert retrieved is None

    def test_get_validators_for_resource(
        self,
        isolated_legacy_validator_registry: None,
        create_validator_metadata: Callable[..., LegacyValidatorMetadata],
    ) -> None:
        """Test retrieving validators for a specific resource type"""
        # Register validators for different resource types
        metadata1 = create_validator_metadata(
            identifier="samplestore_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            description="Sample store validator",
        )

        metadata2 = create_validator_metadata(
            identifier="operation_validator",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_field_paths=["config.field2"],
            description="Operation validator",
        )

        LegacyValidatorRegistry.register(metadata1)
        LegacyValidatorRegistry.register(metadata2)

        # Get validators for SAMPLESTORE
        samplestore_validators = LegacyValidatorRegistry.get_validators_for_resource(
            CoreResourceKinds.SAMPLESTORE
        )
        assert len(samplestore_validators) == 1
        assert samplestore_validators[0].identifier == "samplestore_validator"

        # Get validators for OPERATION
        operation_validators = LegacyValidatorRegistry.get_validators_for_resource(
            CoreResourceKinds.OPERATION
        )
        assert len(operation_validators) == 1
        assert operation_validators[0].identifier == "operation_validator"

    def test_find_validators_for_deprecated_field_paths(
        self,
        isolated_legacy_validator_registry: None,
        create_validator_metadata: Callable[..., LegacyValidatorMetadata],
    ) -> None:
        """Test finding validators that handle specific field paths"""
        # Register validators with different field paths
        metadata1 = create_validator_metadata(
            identifier="validator1",
            deprecated_field_paths=["config.field1", "config.field2"],
            description="Validator 1",
        )

        metadata2 = create_validator_metadata(
            identifier="validator2",
            deprecated_field_paths=["config.specification.field3"],
            description="Validator 2",
        )

        metadata3 = create_validator_metadata(
            identifier="validator3",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_field_paths=["config.properties"],
            description="Validator 3",
        )

        LegacyValidatorRegistry.register(metadata1)
        LegacyValidatorRegistry.register(metadata2)
        LegacyValidatorRegistry.register(metadata3)

        # Find validators for single full path
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.field1"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "validator1"

        # Find validators for nested path
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.specification.field3"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "validator2"

        # Find validators for multiple paths
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE,
            {"config.field1", "config.specification.field3"},
        )
        assert len(validators) == 2
        validator_ids = {v.identifier for v in validators}
        assert validator_ids == {"validator1", "validator2"}

        # Find validators for non-existent path
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.nonexistent"}
        )
        assert len(validators) == 0

        # Verify it doesn't match on leaf names alone (more specific than find_validators_for_fields)
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"field1"}  # Just leaf name, not full path
        )
        assert len(validators) == 0

        # Verify resource type filtering works
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.DISCOVERYSPACE, {"config.properties"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "validator3"

    def test_list_all(
        self,
        isolated_legacy_validator_registry: None,
        create_validator_metadata: Callable[..., LegacyValidatorMetadata],
    ) -> None:
        """Test listing all validators"""
        metadata1 = create_validator_metadata(
            identifier="validator1",
            description="Validator 1",
        )

        metadata2 = create_validator_metadata(
            identifier="validator2",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_field_paths=["config.field2"],
            description="Validator 2",
        )

        LegacyValidatorRegistry.register(metadata1)
        LegacyValidatorRegistry.register(metadata2)

        all_validators = LegacyValidatorRegistry.list_all()
        assert len(all_validators) == 2

    def test_field_path_matching_with_real_validators(
        self, legacy_validators_loaded: None
    ) -> None:
        """Integration test: verify field path matching works with real validators"""

        # Test 1: discoveryspace properties field should match the properties_field_removal validator
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.DISCOVERYSPACE, {"config.properties"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "discoveryspace_properties_field_removal" in validator_ids

        # Test 2: operation actuators field should match the actuators_field_removal validator
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.OPERATION, {"config.actuators"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "operation_actuators_field_removal" in validator_ids

        # Test 3: operation parameters.mode should match randomwalk validator
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.OPERATION, {"config.parameters.mode"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "randomwalk_mode_to_sampler_config" in validator_ids

        # Test 4: samplestore config.specification.module.moduleType should match the module_type validator
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.specification.module.moduleType"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "samplestore_module_type_entitysource_to_samplestore" in validator_ids

        # Test 5: samplestore kind field should match the kind validator
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"kind"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "samplestore_kind_entitysource_to_samplestore" in validator_ids

        # Test 6: Multiple paths should return multiple validators
        validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
            CoreResourceKinds.SAMPLESTORE,
            {
                "config.specification.module.moduleType",
                "config.specification.module.moduleClass",
                "config.specification.module.moduleName",
            },
        )
        assert len(validators) >= 3
        validator_ids = {v.identifier for v in validators}
        assert "samplestore_module_type_entitysource_to_samplestore" in validator_ids
        assert "samplestore_module_class_entitysource_to_samplestore" in validator_ids
        assert "samplestore_module_name_entitysource_to_samplestore" in validator_ids


class TestLegacyValidatorDecorator:
    """Test the @legacy_validator decorator"""

    def test_decorator_registers_validator(
        self, isolated_legacy_validator_registry: None
    ) -> None:
        """Test that the decorator registers the validator"""

        @legacy_validator(
            identifier="test_decorator_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test decorator validator",
        )
        def my_validator(data: dict) -> dict:
            return data

        # Check that validator was registered
        assert len(LegacyValidatorRegistry._validators) == 1
        assert "test_decorator_validator" in LegacyValidatorRegistry._validators

        # Check that the function still works
        test_data = {"key": "value"}
        result = my_validator(test_data)
        assert result == test_data

    def test_decorator_preserves_function_metadata(
        self, isolated_legacy_validator_registry: None
    ) -> None:
        """Test that the decorator preserves function metadata"""

        @legacy_validator(
            identifier="test_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test validator",
        )
        def my_validator(data: dict) -> dict:
            """My validator docstring"""
            return data

        # Check that function name and docstring are preserved
        assert my_validator.__name__ == "my_validator"
        assert my_validator.__doc__ == "My validator docstring"

    def test_validator_function_execution(self) -> None:
        """Test that the validator function executes correctly"""

        @legacy_validator(
            identifier="transform_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Transform validator",
        )
        def transform_validator(data: dict) -> dict:
            if "old_field" in data:
                data["new_field"] = data.pop("old_field")
            return data

        # Test the validator function
        test_data = {"old_field": "value"}
        result = transform_validator(test_data)
        assert "old_field" not in result
        assert result["new_field"] == "value"

        # Verify it was registered correctly
        metadata = LegacyValidatorRegistry.get_validator("transform_validator")
        assert metadata is not None
        # The validator function should be callable and work correctly
        test_data2 = {"old_field": "another_value"}
        result2 = metadata.validator_function(test_data2)
        assert "old_field" not in result2
        assert result2["new_field"] == "another_value"

    # Made with Bob
