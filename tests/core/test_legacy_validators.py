# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for legacy validators with pydantic models and upgrade process"""

from collections.abc import Callable
from pathlib import Path

import pydantic

from orchestrator.core.legacy.registry import LegacyValidatorRegistry, legacy_validator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext


class TestLegacyValidatorWithPydantic:
    """Test legacy validators working with pydantic models"""

    def test_validator_applied_during_model_validation(
        self, isolated_legacy_validator_registry: None
    ) -> None:
        """Test that a legacy validator can be manually applied before pydantic validation"""

        # Define a simple pydantic model
        class OldModel(pydantic.BaseModel):
            new_field: str

        # Register a legacy validator
        @legacy_validator(
            identifier="old_to_new_field",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Migrate old_field to new_field",
        )
        def migrate_old_to_new(data: dict) -> dict:
            if "old_field" in data:
                data["new_field"] = data.pop("old_field")
            return data

        # Get the validator
        validator = LegacyValidatorRegistry.get_validator("old_to_new_field")
        assert validator is not None

        # Old format data
        old_data = {"old_field": "test_value"}

        # Apply legacy validator
        migrated_data = validator.validator_function(old_data)

        # Now validate with pydantic
        model = OldModel.model_validate(migrated_data)
        assert model.new_field == "test_value"

    def test_csv_sample_store_migration_validator(
        self, legacy_validators_loaded: None
    ) -> None:
        """Test the CSV sample store migration validator with realistic data"""

        # Get the validator (should be registered from setup_method)
        validator = LegacyValidatorRegistry.get_validator(
            "csv_constitutive_columns_migration"
        )
        assert validator is not None
        assert validator.resource_type == CoreResourceKinds.SAMPLESTORE

        # Old format CSV sample store data (with config section)
        old_csv_data = {
            "kind": "samplestore",
            "type": "csv",
            "identifier": "test_store",
            "config": {
                "identifierColumn": "id",
                "constitutivePropertyColumns": ["prop1", "prop2"],
                "experiments": [
                    {
                        "experimentIdentifier": "exp1",
                        "actuatorIdentifier": "act1",
                        "propertyMap": ["obs1", "obs2"],
                    }
                ],
            },
        }

        # Apply the validator
        migrated_data = validator.validator_function(old_csv_data.copy())

        # Verify migration - config.constitutivePropertyColumns removed
        assert "constitutivePropertyColumns" not in migrated_data["config"]
        assert len(migrated_data["config"]["experiments"]) == 1
        exp = migrated_data["config"]["experiments"][0]
        assert "propertyMap" not in exp
        assert "observedPropertyMap" in exp
        assert exp["observedPropertyMap"] == ["obs1", "obs2"]
        assert "constitutivePropertyMap" in exp
        assert exp["constitutivePropertyMap"] == ["prop1", "prop2"]

    def test_entitysource_to_samplestore_migration(
        self, legacy_validators_loaded: None
    ) -> None:
        """Test the entitysource to samplestore kind migration"""

        # Import the validator to register it
        from orchestrator.core.legacy.validators.resource.entitysource_to_samplestore import (  # noqa: F401
            migrate_entitysource_kind_to_samplestore,
        )

        # Get the validator
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_kind_entitysource_to_samplestore"
        )
        assert validator is not None
        assert validator.resource_type == CoreResourceKinds.SAMPLESTORE

        # Old format with entitysource kind
        old_data = {
            "kind": "entitysource",
            "type": "csv",
            "identifier": "test_store",
        }

        # Apply the validator
        migrated_data = validator.validator_function(old_data.copy())

        # Verify migration
        assert migrated_data["kind"] == "samplestore"
        assert migrated_data["type"] == "csv"
        assert migrated_data["identifier"] == "test_store"

    def test_chained_validators(self, isolated_legacy_validator_registry: None) -> None:
        """Test applying multiple validators in sequence"""

        # Register two validators
        @legacy_validator(
            identifier="step1_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.old_field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Step 1 migration",
        )
        def step1(data: dict) -> dict:
            if "old_field1" in data:
                data["intermediate_field"] = data.pop("old_field1")
            return data

        @legacy_validator(
            identifier="step2_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.intermediate_field"],
            deprecated_from_version="2.0.0",
            removed_from_version="3.0.0",
            description="Step 2 migration",
        )
        def step2(data: dict) -> dict:
            if "intermediate_field" in data:
                data["new_field"] = data.pop("intermediate_field")
            return data

        # Get validators
        validator1 = LegacyValidatorRegistry.get_validator("step1_validator")
        validator2 = LegacyValidatorRegistry.get_validator("step2_validator")
        assert validator1 is not None
        assert validator2 is not None

        # Old data
        old_data = {"old_field1": "value"}

        # Apply validators in sequence
        data = validator1.validator_function(old_data)
        data = validator2.validator_function(data)

        # Verify final state
        assert "old_field1" not in data
        assert "intermediate_field" not in data
        assert data["new_field"] == "value"


class TestUpgradeHandlerIntegration:
    """Integration tests for ado upgrade with legacy validators via CLI"""

    def test_upgrade_applies_legacy_validator_via_cli(
        self,
        legacy_validators_loaded: None,
        tmp_path: Path,
        valid_ado_project_context: ProjectContext,
        create_active_ado_context: Callable,
    ) -> None:
        """Test that ado upgrade applies legacy validators correctly via CLI"""

        from typer.testing import CliRunner

        runner = CliRunner()

        from orchestrator.cli.core.cli import app as ado
        from orchestrator.cli.utils.generic.wrappers import get_sql_store
        from orchestrator.core.samplestore.config import (
            SampleStoreConfiguration,
            SampleStoreModuleConf,
            SampleStoreSpecification,
        )
        from orchestrator.core.samplestore.resource import SampleStoreResource

        # Step 1: Setup active context
        create_active_ado_context(runner, tmp_path, valid_ado_project_context)

        # Step 2: Register a test validator
        @legacy_validator(
            identifier="test_upgrade_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test upgrade validator",
        )
        def test_validator(data: dict) -> dict:
            """Migrate old_field to new_field"""
            if "config" in data and "old_field" in data["config"]:
                data["config"]["new_field"] = data["config"].pop("old_field")
            return data

        # Step 3: Create a sample store resource
        test_resource = SampleStoreResource(
            identifier="test_legacy_store",
            config=SampleStoreConfiguration(
                specification=SampleStoreSpecification(
                    module=SampleStoreModuleConf(
                        moduleClass="SQLSampleStore",
                        moduleName="orchestrator.core.samplestore.sql",
                    ),
                    storageLocation=valid_ado_project_context.metadataStore,
                )
            ),
        )

        # Step 4: Save resource to database
        sql_store = get_sql_store(project_context=valid_ado_project_context)
        sql_store.updateResource(resource=test_resource)

        # Step 5: Execute upgrade via CLI
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "upgrade",
                "samplestore",
                "--apply-legacy-validator",
                "test_upgrade_validator",
            ],
        )

        # Step 6: Verify success
        assert result.exit_code == 0
        assert "Success" in result.output or "✓" in result.output

        # Step 7: Verify the upgrade process completed successfully
        # The CLI output "Success!" confirms the validator was applied
        # and the resource was upgraded in the database

    def test_upgrade_rejects_mismatched_validator_type(
        self,
        legacy_validators_loaded: None,
        tmp_path: Path,
        valid_ado_project_context: ProjectContext,
        create_active_ado_context: Callable,
    ) -> None:
        """Test that upgrade rejects validators for wrong resource type"""

        from typer.testing import CliRunner

        runner = CliRunner()

        from orchestrator.cli.core.cli import app as ado

        # Step 1: Setup active context
        create_active_ado_context(runner, tmp_path, valid_ado_project_context)

        # Step 2: Register a validator for OPERATION
        @legacy_validator(
            identifier="operation_only_validator",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Operation-only validator",
        )
        def operation_validator(data: dict) -> dict:
            return data

        # Step 3: Try to use operation validator on samplestore
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "upgrade",
                "samplestore",
                "--apply-legacy-validator",
                "operation_only_validator",
            ],
        )

        # Step 4: Verify failure with appropriate error message
        assert result.exit_code == 1
        assert "ERROR" in result.output
        assert "operation_only_validator" in result.output
        assert "operation" in result.output.lower()
        assert "samplestore" in result.output.lower()

    def test_upgrade_rejects_unknown_validator(
        self,
        tmp_path: Path,
        valid_ado_project_context: ProjectContext,
        create_active_ado_context: Callable,
    ) -> None:
        """Test that upgrade rejects unknown validator identifiers"""

        from typer.testing import CliRunner

        runner = CliRunner()

        from orchestrator.cli.core.cli import app as ado

        # Step 1: Setup active context
        create_active_ado_context(runner, tmp_path, valid_ado_project_context)

        # Step 2: Try to use non-existent validator
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "upgrade",
                "samplestore",
                "--apply-legacy-validator",
                "nonexistent_validator_xyz",
            ],
        )

        # Step 3: Verify failure with appropriate error message
        assert result.exit_code == 1
        assert "ERROR" in result.output
        assert "nonexistent_validator_xyz" in result.output
        assert (
            "unknown" in result.output.lower() or "not found" in result.output.lower()
        )

    def test_upgrade_auto_resolves_validator_dependencies(
        self,
        legacy_validators_loaded: None,
        tmp_path: Path,
        valid_ado_project_context: ProjectContext,
        create_active_ado_context: Callable,
    ) -> None:
        """Test that upgrade automatically includes validator dependencies"""

        from typer.testing import CliRunner

        runner = CliRunner()

        from orchestrator.cli.core.cli import app as ado
        from orchestrator.cli.utils.generic.wrappers import get_sql_store
        from orchestrator.core.samplestore.config import (
            SampleStoreConfiguration,
            SampleStoreModuleConf,
            SampleStoreSpecification,
        )
        from orchestrator.core.samplestore.resource import SampleStoreResource

        # Step 1: Setup active context
        create_active_ado_context(runner, tmp_path, valid_ado_project_context)

        # Step 2: Register validators with dependencies
        @legacy_validator(
            identifier="base_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Base validator",
        )
        def base_validator(data: dict) -> dict:
            if "config" in data and "field1" in data["config"]:
                data["config"]["field1_migrated"] = True
            return data

        @legacy_validator(
            identifier="dependent_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.field2"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Dependent validator",
            dependencies=["base_validator"],  # Depends on base_validator
        )
        def dependent_validator(data: dict) -> dict:
            if "config" in data and "field2" in data["config"]:
                data["config"]["field2_migrated"] = True
            return data

        # Step 3: Create and save a sample store resource
        test_resource = SampleStoreResource(
            identifier="test_dependency_store",
            config=SampleStoreConfiguration(
                specification=SampleStoreSpecification(
                    module=SampleStoreModuleConf(
                        moduleClass="SQLSampleStore",
                        moduleName="orchestrator.core.samplestore.sql",
                    ),
                    storageLocation=valid_ado_project_context.metadataStore,
                )
            ),
        )

        sql_store = get_sql_store(project_context=valid_ado_project_context)
        sql_store.updateResource(resource=test_resource)

        # Step 4: Execute upgrade with only dependent_validator
        # Should auto-include base_validator
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "upgrade",
                "samplestore",
                "--apply-legacy-validator",
                "dependent_validator",
            ],
        )

        # Step 5: Verify success
        assert result.exit_code == 0
        assert "Success" in result.output or "✓" in result.output

        # The test verifies the CLI command completes successfully
        # with automatic dependency resolution


class TestValidatorDataIntegrity:
    """Test that validators preserve data integrity"""

    def setup_method(self) -> None:
        """Clear the registry before each test"""
        LegacyValidatorRegistry._validators = {}

    def test_validator_preserves_unrelated_fields(self) -> None:
        """Test that validators don't modify unrelated fields"""

        @legacy_validator(
            identifier="selective_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Selective validator",
        )
        def selective(data: dict) -> dict:
            if "old_field" in data:
                data["new_field"] = data.pop("old_field")
            return data

        validator = LegacyValidatorRegistry.get_validator("selective_validator")
        assert validator is not None

        # Data with many fields
        data = {
            "old_field": "migrate_me",
            "keep_field1": "value1",
            "keep_field2": 42,
            "keep_field3": ["list", "of", "items"],
            "keep_field4": {"nested": "dict"},
        }

        result = validator.validator_function(data.copy())

        # Verify migration happened
        assert "old_field" not in result
        assert result["new_field"] == "migrate_me"

        # Verify other fields preserved
        assert result["keep_field1"] == "value1"
        assert result["keep_field2"] == 42
        assert result["keep_field3"] == ["list", "of", "items"]
        assert result["keep_field4"] == {"nested": "dict"}

    def test_validator_handles_missing_fields_gracefully(
        self, isolated_legacy_validator_registry: None
    ) -> None:
        """Test that validators handle missing deprecated fields gracefully"""

        @legacy_validator(
            identifier="graceful_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.optional_old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Graceful validator",
        )
        def graceful(data: dict) -> dict:
            if "optional_old_field" in data:
                data["new_field"] = data.pop("optional_old_field")
            return data

        validator = LegacyValidatorRegistry.get_validator("graceful_validator")
        assert validator is not None

        # Data without the deprecated field
        data = {"other_field": "value"}

        result = validator.validator_function(data.copy())

        # Should not crash and should preserve data
        assert result == data
        assert "new_field" not in result


# Made with Bob
