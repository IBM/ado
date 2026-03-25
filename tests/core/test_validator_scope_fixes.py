# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for Phase 1 validator scope fixes - verifying validators only operate on config level"""

import importlib

from orchestrator.core.legacy.registry import LegacyValidatorRegistry


class TestValidatorScopeFixes:
    """Test that validators correctly operate only on config level after Phase 1 fixes"""

    def setup_method(self) -> None:
        """Clear registry and reload validators before each test for isolation"""
        # Clear the registry to ensure clean state
        LegacyValidatorRegistry._validators = {}
        # Reload all validator module files to re-register them
        import orchestrator.core.legacy.validators.discoveryspace.entitysource_to_samplestore
        import orchestrator.core.legacy.validators.discoveryspace.properties_field_removal
        import orchestrator.core.legacy.validators.operation.actuators_field_removal
        import orchestrator.core.legacy.validators.resource.entitysource_to_samplestore
        import orchestrator.core.legacy.validators.samplestore.entitysource_migrations
        import orchestrator.core.legacy.validators.samplestore.v1_to_v2_csv_migration

        importlib.reload(
            orchestrator.core.legacy.validators.discoveryspace.properties_field_removal
        )
        importlib.reload(
            orchestrator.core.legacy.validators.discoveryspace.entitysource_to_samplestore
        )
        importlib.reload(
            orchestrator.core.legacy.validators.operation.actuators_field_removal
        )
        importlib.reload(
            orchestrator.core.legacy.validators.resource.entitysource_to_samplestore
        )
        importlib.reload(
            orchestrator.core.legacy.validators.samplestore.v1_to_v2_csv_migration
        )
        importlib.reload(
            orchestrator.core.legacy.validators.samplestore.entitysource_migrations
        )

    def test_discoveryspace_properties_removal_scope(self) -> None:
        """Verify properties_field_removal only modifies config, not resource level"""

        # Test data with 'properties' at both levels
        resource_data = {
            "kind": "discoveryspace",
            "identifier": "test-space",
            "properties": "SHOULD_NOT_BE_REMOVED",  # Resource level
            "config": {
                "properties": ["prop1", "prop2"],  # Config level (should be removed)
                "sampleStoreIdentifier": "store-1",
            },
        }

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "discoveryspace_properties_field_removal"
        )
        assert validator is not None

        # Apply validator
        result = validator.validator_function(resource_data.copy())

        # Verify: resource-level field unchanged, config-level field removed
        assert "properties" in result  # Resource level preserved
        assert result["properties"] == "SHOULD_NOT_BE_REMOVED"
        assert "properties" not in result["config"]  # Config level removed
        assert result["config"]["sampleStoreIdentifier"] == "store-1"

    def test_discoveryspace_entitysource_migration_scope(self) -> None:
        """Verify entitysource_to_samplestore only modifies config, not resource level"""

        # Test data with entitySourceIdentifier at both levels
        resource_data = {
            "kind": "discoveryspace",
            "identifier": "test-space",
            "entitySourceIdentifier": "SHOULD_NOT_BE_REMOVED",  # Resource level
            "config": {
                "entitySourceIdentifier": "old-source",  # Config level (should migrate)
            },
        }

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "discoveryspace_entitysource_to_samplestore"
        )
        assert validator is not None

        # Apply validator
        result = validator.validator_function(resource_data.copy())

        # Verify: resource-level field unchanged, config-level field migrated
        assert "entitySourceIdentifier" in result  # Resource level preserved
        assert result["entitySourceIdentifier"] == "SHOULD_NOT_BE_REMOVED"
        assert "entitySourceIdentifier" not in result["config"]  # Config level removed
        assert result["config"]["sampleStoreIdentifier"] == "old-source"  # Migrated

    def test_operation_actuators_removal_scope(self) -> None:
        """Verify actuators_field_removal only modifies config, not resource level"""

        # Test data with actuators at both levels
        resource_data = {
            "kind": "operation",
            "identifier": "test-op",
            "actuators": "SHOULD_NOT_BE_REMOVED",  # Resource level
            "config": {
                "actuators": ["act1", "act2"],  # Config level (should be removed)
                "operatorIdentifier": "op1",
            },
        }

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "operation_actuators_field_removal"
        )
        assert validator is not None

        # Apply validator
        result = validator.validator_function(resource_data.copy())

        # Verify: resource-level field unchanged, config-level field removed
        assert "actuators" in result  # Resource level preserved
        assert result["actuators"] == "SHOULD_NOT_BE_REMOVED"
        assert "actuators" not in result["config"]  # Config level removed
        assert result["config"]["operatorIdentifier"] == "op1"

    def test_samplestore_module_type_entitysource_migration_scope(self) -> None:
        """Verify entitysource module type migration only modifies config, not resource level"""

        # Test data with moduleType at both levels
        resource_data = {
            "kind": "samplestore",
            "type": "csv",
            "identifier": "test-store",
            "moduleType": "SHOULD_NOT_BE_REMOVED",  # Resource level
            "config": {
                "moduleType": "entity_source",  # Config level (should migrate)
            },
        }

        # Get validator for module type entitysource migration
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_module_type_entitysource_to_samplestore"
        )
        assert validator is not None

        # Apply validator
        result = validator.validator_function(resource_data.copy())

        # Verify: resource-level field unchanged, config-level field migrated
        assert "moduleType" in result  # Resource level preserved
        assert result["moduleType"] == "SHOULD_NOT_BE_REMOVED"
        assert result["config"]["moduleType"] == "sample_store"  # Migrated

    def test_samplestore_csv_migration_scope(self) -> None:
        """Verify CSV v1 to v2 migration only modifies config, not resource level"""

        # Test data with constitutivePropertyColumns at both levels
        resource_data = {
            "kind": "samplestore",
            "type": "csv",
            "identifier": "test-store",
            "constitutivePropertyColumns": "SHOULD_NOT_BE_REMOVED",  # Resource level
            "config": {
                "identifierColumn": "id",
                "constitutivePropertyColumns": ["prop1", "prop2"],  # Config (migrate)
                "experiments": [
                    {
                        "experimentIdentifier": "exp1",
                        "actuatorIdentifier": "act1",
                        "propertyMap": ["obs1", "obs2"],
                    }
                ],
            },
        }

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "csv_constitutive_columns_migration"
        )
        assert validator is not None

        # Apply validator
        result = validator.validator_function(resource_data.copy())

        # Verify: resource-level field unchanged, config-level field migrated
        assert "constitutivePropertyColumns" in result  # Resource level preserved
        assert result["constitutivePropertyColumns"] == "SHOULD_NOT_BE_REMOVED"
        assert "constitutivePropertyColumns" not in result["config"]  # Config removed
        # Verify migration happened in config
        exp = result["config"]["experiments"][0]
        assert "propertyMap" not in exp
        assert "observedPropertyMap" in exp
        assert exp["observedPropertyMap"] == ["obs1", "obs2"]
        assert "constitutivePropertyMap" in exp
        assert exp["constitutivePropertyMap"] == ["prop1", "prop2"]

    def test_resource_kind_field_operates_at_resource_level(self) -> None:
        """Verify resource-level validators (like kind migration) operate at resource level"""

        # Test data with entitysource kind at resource level
        resource_data = {
            "kind": "entitysource",  # Resource level (should be migrated)
            "type": "csv",
            "identifier": "test-store",
            "config": {
                "identifierColumn": "id",
            },
        }

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_kind_entitysource_to_samplestore"
        )
        assert validator is not None

        # Apply validator
        result = validator.validator_function(resource_data.copy())

        # Verify: resource-level kind field was migrated
        assert result["kind"] == "samplestore"
        assert result["type"] == "csv"
        assert result["identifier"] == "test-store"

    def test_validators_preserve_unrelated_fields(self) -> None:
        """Verify validators don't modify unrelated fields at any level"""

        # Test data with many fields
        resource_data = {
            "kind": "discoveryspace",
            "identifier": "test-space",
            "unrelated_resource_field": "preserve_me",
            "config": {
                "properties": ["prop1", "prop2"],  # Will be removed
                "sampleStoreIdentifier": "store-1",
                "unrelated_config_field": "preserve_me_too",
                "nested": {
                    "deep_field": "also_preserve",
                },
            },
        }

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "discoveryspace_properties_field_removal"
        )
        assert validator is not None

        # Apply validator
        result = validator.validator_function(resource_data.copy())

        # Verify: unrelated fields preserved at all levels
        assert result["unrelated_resource_field"] == "preserve_me"
        assert result["config"]["unrelated_config_field"] == "preserve_me_too"
        assert result["config"]["nested"]["deep_field"] == "also_preserve"
        assert result["config"]["sampleStoreIdentifier"] == "store-1"
        # But deprecated field removed
        assert "properties" not in result["config"]

    def test_validators_handle_missing_config_gracefully(self) -> None:
        """Verify validators handle missing config field gracefully"""

        # Test data without config
        resource_data = {
            "kind": "discoveryspace",
            "identifier": "test-space",
        }

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "discoveryspace_properties_field_removal"
        )
        assert validator is not None

        # Apply validator - should not crash
        result = validator.validator_function(resource_data.copy())

        # Verify: data unchanged
        assert result == resource_data

    def test_validators_handle_empty_config_gracefully(self) -> None:
        """Verify validators handle empty config dict gracefully"""

        # Test data with empty config
        resource_data = {
            "kind": "discoveryspace",
            "identifier": "test-space",
            "config": {},
        }

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "discoveryspace_properties_field_removal"
        )
        assert validator is not None

        # Apply validator - should not crash
        result = validator.validator_function(resource_data.copy())

        # Verify: data unchanged
        assert result == resource_data

    def test_samplestore_module_name_exact_match(self) -> None:
        """Verify module name migration uses exact matching only"""

        # Get validator
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_module_name_entitysource_to_samplestore"
        )
        assert validator is not None

        # Test data with module names that should NOT be migrated
        # (not exact matches)
        no_migration_cases = [
            "my_orchestrator.core.entitysource_wrapper",  # Contains substring but not exact
            "orchestrator.core.entitysource.csv",  # Submodule, not exact match
            "orchestrator.plugins.entitysources.custom",  # Submodule, not exact match
        ]

        for module_name in no_migration_cases:
            data = {
                "kind": "samplestore",
                "type": "csv",
                "identifier": "test-store",
                "config": {
                    "moduleName": module_name,
                },
            }
            result = validator.validator_function(data.copy())
            assert result["config"]["moduleName"] == module_name, (
                f"Expected {module_name} to NOT be migrated, "
                f"but got {result['config']['moduleName']}"
            )

        # Test exact matches that SHOULD migrate
        exact_match_cases = [
            (
                "orchestrator.core.entitysource",
                "orchestrator.core.samplestore",
            ),
            (
                "orchestrator.plugins.entitysources",
                "orchestrator.plugins.samplestores",
            ),
        ]

        for old_name, expected_new_name in exact_match_cases:
            data = {
                "kind": "samplestore",
                "type": "csv",
                "identifier": "test-store",
                "config": {
                    "moduleName": old_name,
                },
            }
            result = validator.validator_function(data.copy())
            assert result["config"]["moduleName"] == expected_new_name, (
                f"Expected {old_name} to migrate to {expected_new_name}, "
                f"but got {result['config']['moduleName']}"
            )


# Made with Bob
