# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for the legacy validator registry"""

from orchestrator.core.legacy.metadata import LegacyValidatorMetadata
from orchestrator.core.legacy.registry import (
    LegacyValidatorRegistry,
    legacy_validator,
)
from orchestrator.core.resources import CoreResourceKinds


class TestLegacyValidatorMetadata:
    """Test the LegacyValidatorMetadata model"""

    def test_create_metadata(self) -> None:
        """Test creating validator metadata"""

        def dummy_validator(data: dict) -> dict:
            return data

        metadata = LegacyValidatorMetadata(
            identifier="test_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1", "field2"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test validator",
            validator_function=dummy_validator,
        )

        assert metadata.identifier == "test_validator"
        assert metadata.resource_type == CoreResourceKinds.SAMPLESTORE
        assert metadata.deprecated_fields == ["field1", "field2"]
        assert metadata.deprecated_from_version == "1.0.0"
        assert metadata.removed_from_version == "2.0.0"
        assert metadata.description == "Test validator"
        assert metadata.validator_function == dummy_validator

    def test_metadata_serialization(self) -> None:
        """Test that validator function is excluded from serialization"""

        def dummy_validator(data: dict) -> dict:
            return data

        metadata = LegacyValidatorMetadata(
            identifier="test_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test validator",
            validator_function=dummy_validator,
        )

        # Serialize to dict
        data = metadata.model_dump()

        # validator_function should be excluded
        assert "validator_function" not in data
        assert "identifier" in data
        assert "resource_type" in data


class TestLegacyValidatorRegistry:
    """Test the LegacyValidatorRegistry class"""

    def setup_method(self) -> None:
        """Clear the registry before each test"""
        LegacyValidatorRegistry._validators = {}

    def test_register_validator(self) -> None:
        """Test registering a validator"""

        def dummy_validator(data: dict) -> dict:
            return data

        metadata = LegacyValidatorMetadata(
            identifier="test_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test validator",
            validator_function=dummy_validator,
        )

        LegacyValidatorRegistry.register(metadata)

        assert len(LegacyValidatorRegistry._validators) == 1
        assert "test_validator" in LegacyValidatorRegistry._validators

    def test_get_validator(self) -> None:
        """Test retrieving a validator by identifier"""

        def dummy_validator(data: dict) -> dict:
            return data

        metadata = LegacyValidatorMetadata(
            identifier="test_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test validator",
            validator_function=dummy_validator,
        )

        LegacyValidatorRegistry.register(metadata)

        retrieved = LegacyValidatorRegistry.get_validator("test_validator")
        assert retrieved is not None
        assert retrieved.identifier == "test_validator"

    def test_get_nonexistent_validator(self) -> None:
        """Test retrieving a validator that doesn't exist"""
        retrieved = LegacyValidatorRegistry.get_validator("nonexistent")
        assert retrieved is None

    def test_get_validators_for_resource(self) -> None:
        """Test retrieving validators for a specific resource type"""

        def dummy_validator(data: dict) -> dict:
            return data

        # Register validators for different resource types
        metadata1 = LegacyValidatorMetadata(
            identifier="samplestore_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Sample store validator",
            validator_function=dummy_validator,
        )

        metadata2 = LegacyValidatorMetadata(
            identifier="operation_validator",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_fields=["field2"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Operation validator",
            validator_function=dummy_validator,
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

    def test_find_validators_for_fields(self) -> None:
        """Test finding validators that handle specific fields"""

        def dummy_validator(data: dict) -> dict:
            return data

        # Register validators with different fields
        metadata1 = LegacyValidatorMetadata(
            identifier="validator1",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1", "field2"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator 1",
            validator_function=dummy_validator,
        )

        metadata2 = LegacyValidatorMetadata(
            identifier="validator2",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field3", "field4"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator 2",
            validator_function=dummy_validator,
        )

        LegacyValidatorRegistry.register(metadata1)
        LegacyValidatorRegistry.register(metadata2)

        # Find validators for field1
        validators = LegacyValidatorRegistry.find_validators_for_fields(
            CoreResourceKinds.SAMPLESTORE, ["field1"]
        )
        assert len(validators) == 1
        assert validators[0].identifier == "validator1"

        # Find validators for field3
        validators = LegacyValidatorRegistry.find_validators_for_fields(
            CoreResourceKinds.SAMPLESTORE, ["field3"]
        )
        assert len(validators) == 1
        assert validators[0].identifier == "validator2"

        # Find validators for multiple fields
        validators = LegacyValidatorRegistry.find_validators_for_fields(
            CoreResourceKinds.SAMPLESTORE, ["field1", "field3"]
        )
        assert len(validators) == 2

        # Find validators for nonexistent field
        validators = LegacyValidatorRegistry.find_validators_for_fields(
            CoreResourceKinds.SAMPLESTORE, ["nonexistent"]
        )
        assert len(validators) == 0

    def test_find_validators_for_field_paths(self) -> None:
        """Test finding validators that handle specific field paths"""

        def dummy_validator(data: dict) -> dict:
            return data

        # Register validators with different field paths
        metadata1 = LegacyValidatorMetadata(
            identifier="validator1",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1", "field2"],
            field_paths=["config.field1", "config.field2"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator 1",
            validator_function=dummy_validator,
        )

        metadata2 = LegacyValidatorMetadata(
            identifier="validator2",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field3"],
            field_paths=["config.specification.field3"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator 2",
            validator_function=dummy_validator,
        )

        metadata3 = LegacyValidatorMetadata(
            identifier="validator3",
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
            deprecated_fields=["properties"],
            field_paths=["config.properties"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator 3",
            validator_function=dummy_validator,
        )

        LegacyValidatorRegistry.register(metadata1)
        LegacyValidatorRegistry.register(metadata2)
        LegacyValidatorRegistry.register(metadata3)

        # Find validators for single full path
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.field1"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "validator1"

        # Find validators for nested path
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.specification.field3"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "validator2"

        # Find validators for multiple paths
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.SAMPLESTORE,
            {"config.field1", "config.specification.field3"},
        )
        assert len(validators) == 2
        validator_ids = {v.identifier for v in validators}
        assert validator_ids == {"validator1", "validator2"}

        # Find validators for non-existent path
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.nonexistent"}
        )
        assert len(validators) == 0

        # Verify it doesn't match on leaf names alone (more specific than find_validators_for_fields)
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"field1"}  # Just leaf name, not full path
        )
        assert len(validators) == 0

        # Verify resource type filtering works
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.DISCOVERYSPACE, {"config.properties"}
        )
        assert len(validators) == 1
        assert validators[0].identifier == "validator3"

    def test_list_all(self) -> None:
        """Test listing all validators"""

        def dummy_validator(data: dict) -> dict:
            return data

        metadata1 = LegacyValidatorMetadata(
            identifier="validator1",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator 1",
            validator_function=dummy_validator,
        )

        metadata2 = LegacyValidatorMetadata(
            identifier="validator2",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_fields=["field2"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Validator 2",
            validator_function=dummy_validator,
        )

        LegacyValidatorRegistry.register(metadata1)
        LegacyValidatorRegistry.register(metadata2)

        all_validators = LegacyValidatorRegistry.list_all()
        assert len(all_validators) == 2

    def test_field_path_matching_with_real_validators(self) -> None:
        """Integration test: verify field path matching works with real validators"""
        # Import validators to trigger registration
        import orchestrator.core.legacy.validators  # noqa: F401

        # Test 1: discoveryspace properties field should match the properties_field_removal validator
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.DISCOVERYSPACE, {"config.properties"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "discoveryspace_properties_field_removal" in validator_ids

        # Test 2: operation actuators field should match the actuators_field_removal validator
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.OPERATION, {"config.actuators"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "operation_actuators_field_removal" in validator_ids

        # Test 3: operation parameters.mode should match randomwalk validator
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.OPERATION, {"config.parameters.mode"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "randomwalk_mode_to_sampler_config" in validator_ids

        # Test 4: samplestore config.moduleType should match the module_type validator
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"config.moduleType"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "samplestore_module_type_entitysource_to_samplestore" in validator_ids

        # Test 5: samplestore kind field should match the kind validator
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.SAMPLESTORE, {"kind"}
        )
        assert len(validators) >= 1
        validator_ids = {v.identifier for v in validators}
        assert "samplestore_kind_entitysource_to_samplestore" in validator_ids

        # Test 6: Multiple paths should return multiple validators
        validators = LegacyValidatorRegistry.find_validators_for_field_paths(
            CoreResourceKinds.SAMPLESTORE,
            {"config.moduleType", "config.moduleClass", "config.moduleName"},
        )
        assert len(validators) >= 3
        validator_ids = {v.identifier for v in validators}
        assert "samplestore_module_type_entitysource_to_samplestore" in validator_ids
        assert "samplestore_module_class_entitysource_to_samplestore" in validator_ids
        assert "samplestore_module_name_entitysource_to_samplestore" in validator_ids


class TestLegacyValidatorDecorator:
    """Test the @legacy_validator decorator"""

    def setup_method(self) -> None:
        """Clear the registry before each test"""
        LegacyValidatorRegistry._validators = {}

    def test_decorator_registers_validator(self) -> None:
        """Test that the decorator registers the validator"""

        @legacy_validator(
            identifier="test_decorator_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1"],
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

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test that the decorator preserves function metadata"""

        @legacy_validator(
            identifier="test_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["field1"],
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
            deprecated_fields=["old_field"],
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
