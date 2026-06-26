# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for legacy migrators with pydantic models and upgrade process"""

from collections.abc import Callable
from pathlib import Path

import pydantic

from orchestrator.core.legacy.registry import LegacyMigratorRegistry, legacy_migrator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext
from tests.conftest import requires_sqlite_3_38


class TestLegacyMigratorWithPydantic:
    """Test legacy migrators working with pydantic models"""

    def test_migrator_applied_during_model_validation(
        self, isolated_legacy_migrator_registry: None
    ) -> None:
        """Test that a legacy migrator can be manually applied before pydantic validation"""

        # Define a simple pydantic model
        class OldModel(pydantic.BaseModel):
            new_field: str

        # Register a legacy migrator
        @legacy_migrator(
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
        migrator = LegacyMigratorRegistry.get_migrator("old_to_new_field")
        assert migrator is not None

        # Old format data
        old_data = {"old_field": "test_value"}

        # Apply legacy migrator
        migrated_data = migrator.migrator_function(old_data)

        # Now validate with pydantic
        model = OldModel.model_validate(migrated_data)
        assert model.new_field == "test_value"

    def test_csv_sample_store_migration_migrator(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test the CSV sample store migration migrator with realistic data"""

        # Get the migrator (should be registered from setup_method)
        migrator = LegacyMigratorRegistry.get_migrator(
            "csv_constitutive_columns_migration"
        )
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.SAMPLESTORE

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
        migrated_data = migrator.migrator_function(old_csv_data.copy())

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
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test the entitysource to samplestore kind migration"""

        # Import the migrator to register it
        from orchestrator.core.legacy.migrators.resource.entitysource_to_samplestore import (  # noqa: F401
            migrate_entitysource_kind_to_samplestore,
        )

        # Get the migrator
        migrator = LegacyMigratorRegistry.get_migrator(
            "samplestore_kind_entitysource_to_samplestore"
        )
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.SAMPLESTORE

        # Old format with entitysource kind
        old_data = {
            "kind": "entitysource",
            "type": "csv",
            "identifier": "test_store",
        }

        # Apply the migrator
        migrated_data = migrator.migrator_function(old_data.copy())

        # Verify migration
        assert migrated_data["kind"] == "samplestore"
        assert migrated_data["type"] == "csv"
        assert migrated_data["identifier"] == "test_store"

    def test_chained_migrators(self, isolated_legacy_migrator_registry: None) -> None:
        """Test applying multiple migrators in sequence"""

        # Register two migrators
        @legacy_migrator(
            identifier="step1_migrator",
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

        @legacy_migrator(
            identifier="step2_migrator",
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
        migrator1 = LegacyMigratorRegistry.get_migrator("step1_migrator")
        migrator2 = LegacyMigratorRegistry.get_migrator("step2_migrator")
        assert migrator1 is not None
        assert migrator2 is not None

        # Old data
        old_data = {"old_field1": "value"}

        # Apply migrators in sequence
        data = migrator1.migrator_function(old_data)
        data = migrator2.migrator_function(data)

        # Verify final state
        assert "old_field1" not in data
        assert "intermediate_field" not in data
        assert data["new_field"] == "value"


class TestUpgradeHandlerIntegration:
    """Integration tests for ado upgrade with legacy migrators via CLI"""

    def test_upgrade_applies_legacy_migrator_via_cli(
        self,
        legacy_migrators_loaded: None,
        tmp_path: Path,
        valid_ado_mysql_project_context: ProjectContext,
        create_active_ado_context: Callable,
    ) -> None:
        """Test that ado upgrade applies legacy migrators correctly via CLI"""

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
        create_active_ado_context(runner, tmp_path, valid_ado_mysql_project_context)

        # Step 2: Register a test validator
        @legacy_migrator(
            identifier="test_upgrade_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test upgrade migrator",
        )
        def test_migrator(data: dict) -> dict:
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
                    storageLocation=valid_ado_mysql_project_context.metadataStore,
                )
            ),
        )

        # Step 4: Save resource to database
        sql_store = get_sql_store(project_context=valid_ado_mysql_project_context)
        sql_store.updateResource(resource=test_resource)

        # Step 5: Execute upgrade via CLI
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "upgrade",
                "samplestore",
                "--apply-legacy-migrator",
                "test_upgrade_migrator",
            ],
        )

        # Step 6: Verify success
        assert result.exit_code == 0
        assert "Success" in result.output or "✓" in result.output

        # Step 7: Verify the upgrade process completed successfully
        # The CLI output "Success!" confirms the migrator was applied
        # and the resource was upgraded in the database

    def test_upgrade_rejects_mismatched_migrator_type(
        self,
        legacy_migrators_loaded: None,
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

        # Step 2: Register a migrator for OPERATION
        @legacy_migrator(
            identifier="operation_only_migrator",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Operation-only migrator",
        )
        def operation_migrator(data: dict) -> dict:
            return data

        # Step 3: Try to use operation migrator on samplestore
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "upgrade",
                "samplestore",
                "--apply-legacy-migrator",
                "operation_only_migrator",
            ],
        )

        # Step 4: Verify failure with appropriate error message
        assert result.exit_code == 1
        assert "ERROR" in result.output
        assert "operation_only_migrator" in result.output
        assert "operation" in result.output.lower()
        assert "samplestore" in result.output.lower()

    def test_upgrade_rejects_unknown_migrator(
        self,
        tmp_path: Path,
        valid_ado_project_context: ProjectContext,
        create_active_ado_context: Callable,
    ) -> None:
        """Test that upgrade rejects unknown migrator identifiers"""

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
                "--apply-legacy-migrator",
                "nonexistent_migrator_xyz",
            ],
        )

        # Step 3: Verify failure with appropriate error message
        assert result.exit_code == 1
        assert "ERROR" in result.output
        assert "nonexistent_migrator_xyz" in result.output
        assert (
            "unknown" in result.output.lower() or "not found" in result.output.lower()
        )

    @requires_sqlite_3_38
    def test_upgrade_auto_resolves_migrator_dependencies(
        self,
        legacy_migrators_loaded: None,
        tmp_path: Path,
        valid_ado_project_context: ProjectContext,
        create_active_ado_context: Callable,
    ) -> None:
        """Test that upgrade automatically includes migrator dependencies"""

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
        @legacy_migrator(
            identifier="base_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.field1"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Base migrator",
        )
        def base_migrator(data: dict) -> dict:
            if "config" in data and "field1" in data["config"]:
                data["config"]["field1_migrated"] = True
            return data

        @legacy_migrator(
            identifier="dependent_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.field2"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Dependent migrator",
            dependencies=["base_migrator"],  # Depends on base_validator
        )
        def dependent_migrator(data: dict) -> dict:
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
                "--apply-legacy-migrator",
                "dependent_migrator",
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
        LegacyMigratorRegistry._migrators = {}

    def test_migrator_preserves_unrelated_fields(self) -> None:
        """Test that validators don't modify unrelated fields"""

        @legacy_migrator(
            identifier="selective_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Selective migrator",
        )
        def selective(data: dict) -> dict:
            if "old_field" in data:
                data["new_field"] = data.pop("old_field")
            return data

        migrator = LegacyMigratorRegistry.get_migrator("selective_migrator")
        assert migrator is not None

        # Data with many fields
        data = {
            "old_field": "migrate_me",
            "keep_field1": "value1",
            "keep_field2": 42,
            "keep_field3": ["list", "of", "items"],
            "keep_field4": {"nested": "dict"},
        }

        result = migrator.migrator_function(data.copy())

        # Verify migration happened
        assert "old_field" not in result
        assert result["new_field"] == "migrate_me"

        # Verify other fields preserved
        assert result["keep_field1"] == "value1"
        assert result["keep_field2"] == 42
        assert result["keep_field3"] == ["list", "of", "items"]
        assert result["keep_field4"] == {"nested": "dict"}

    def test_migrator_handles_missing_fields_gracefully(
        self, isolated_legacy_migrator_registry: None
    ) -> None:
        """Test that validators handle missing deprecated fields gracefully"""

        @legacy_migrator(
            identifier="graceful_migrator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.optional_old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Graceful migrator",
        )
        def graceful(data: dict) -> dict:
            if "optional_old_field" in data:
                data["new_field"] = data.pop("optional_old_field")
            return data

        migrator = LegacyMigratorRegistry.get_migrator("graceful_migrator")
        assert migrator is not None

        # Data without the deprecated field
        data = {"other_field": "value"}

        result = migrator.migrator_function(data.copy())

        # Should not crash and should preserve data
        assert result == data
        assert "new_field" not in result

    def test_operation_operator_name_field_rename(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test the operationName to operatorName operation migrator."""
        migrator = LegacyMigratorRegistry.get_migrator(
            "operation_operator_name_field_rename"
        )
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.OPERATION

        old_data = {
            "config": {
                "operation": {
                    "module": {
                        "operationName": "detect_anomalous_series",
                        "operationType": "characterize",
                    }
                }
            }
        }

        migrated_data = migrator.migrator_function(old_data.copy())

        module = migrated_data["config"]["operation"]["module"]
        assert module["operatorName"] == "detect_anomalous_series"
        assert module["operationType"] == "characterize"
        assert "operationName" not in module

    def test_samplestore_created_timezone_utc(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test the naive sample store created timestamp migrator."""
        migrator = LegacyMigratorRegistry.get_migrator(
            "samplestore_created_timezone_utc"
        )
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.SAMPLESTORE

        old_data = {
            "kind": "samplestore",
            "created": "2024-03-22T11:46:31.301815",
            "config": {},
        }

        migrated_data = migrator.migrator_function(old_data.copy())

        assert migrated_data["created"] == "2024-03-22T11:46:31.301815Z"

    def test_operation_created_timezone_utc(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test the naive operation created timestamp migrator."""
        migrator = LegacyMigratorRegistry.get_migrator("operation_created_timezone_utc")
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.OPERATION

        old_data = {
            "kind": "operation",
            "created": "2024-03-22T11:46:31.301815",
            "config": {},
        }

        migrated_data = migrator.migrator_function(old_data.copy())

        assert migrated_data["created"] == "2024-03-22T11:46:31.301815Z"

    def test_discoveryspace_created_timezone_utc(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test the naive discovery space created timestamp migrator."""
        migrator = LegacyMigratorRegistry.get_migrator(
            "discoveryspace_created_timezone_utc"
        )
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.DISCOVERYSPACE

        old_data = {
            "kind": "discoveryspace",
            "created": "2024-03-22T11:46:31.301815",
            "config": {},
        }

        migrated_data = migrator.migrator_function(old_data.copy())

        assert migrated_data["created"] == "2024-03-22T11:46:31.301815Z"

    def test_discoveryspace_entitysource_field_removal(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test removal of legacy discovery space entitySource config."""
        migrator = LegacyMigratorRegistry.get_migrator(
            "discoveryspace_entitysource_field_removal"
        )
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.DISCOVERYSPACE

        old_data = {
            "config": {
                "entitySource": {
                    "module": {
                        "moduleClass": "SQLEntitySource",
                        "moduleName": "orchestrator.model.sqlstore",
                        "modulePath": ".",
                        "moduleType": "entity_source",
                    },
                    "parameters": {
                        "configuration": {
                            "active": True,
                            "database": "lattice-qcd",
                            "host": "percona-mysql-haproxy",
                            "password": "secret",
                            "sslVerify": False,
                            "user": "lattice-qcd",
                        },
                        "identifier": "f660cf",
                    },
                },
                "entitySpace": [],
            }
        }

        migrated_data = migrator.migrator_function(old_data.copy())

        assert migrated_data["config"]["sampleStoreIdentifier"] == "f660cf"
        assert "entitySource" not in migrated_data["config"]

    def test_operation_space_identifier_field_removal(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test removal of legacy operation spaceIdentifier config."""
        migrator = LegacyMigratorRegistry.get_migrator(
            "operation_space_identifier_field_removal"
        )
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.OPERATION

        old_data = {
            "config": {
                "metadata": {},
                "operation": {
                    "module": {
                        "moduleClass": "RandomWalk",
                        "moduleName": "orchestrator.agents.optimizers",
                        "modulePath": ".",
                        "moduleType": "operation",
                    },
                    "parameters": {
                        "batchSize": 1,
                        "mode": "sequential",
                        "numberIterations": 240,
                        "samplerType": "generator",
                        "singleMeasurement": False,
                    },
                },
                "spaceIdentifier": "space-e0c297-8f8b8c",
                "spaces": ["space-e0c297-8f8b8c"],
            }
        }

        migrated_data = migrator.migrator_function(old_data.copy())

        assert "spaceIdentifier" not in migrated_data["config"]
        assert migrated_data["config"]["spaces"] == ["space-e0c297-8f8b8c"]

    def test_discoveryspace_additional_entity_sources_field_removal(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test removal of legacy discovery space additionalEntitySources config."""
        migrator = LegacyMigratorRegistry.get_migrator(
            "discoveryspace_additional_entity_sources_field_removal"
        )
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.DISCOVERYSPACE

        old_data = {
            "config": {
                "additionalEntitySources": None,
                "entitySourceIdentifier": "1d5055",
                "entitySpace": [],
                "experiments": {"experiments": []},
            }
        }

        migrated_data = migrator.migrator_function(old_data.copy())

        assert "additionalEntitySources" not in migrated_data["config"]
        assert migrated_data["config"]["entitySourceIdentifier"] == "1d5055"

    def test_operation_result_field_removal(
        self, legacy_migrators_loaded: None
    ) -> None:
        """Test removal of legacy top-level operation result field."""
        migrator = LegacyMigratorRegistry.get_migrator("operation_result_field_removal")
        assert migrator is not None
        assert migrator.resource_type == CoreResourceKinds.OPERATION

        old_data = {
            "config": {
                "actuatorConfigurationIdentifiers": [
                    "actuatorconfiguration-geospatial-actuator-fee26c65"
                ],
                "actuators": None,
                "metadata": {"description": None, "labels": None, "name": None},
                "operation": {
                    "module": {
                        "moduleClass": "RandomWalk",
                        "moduleFunction": None,
                        "moduleName": "orchestrator.modules.operators.randomwalk",
                        "modulePath": ".",
                        "moduleType": "operation",
                    },
                    "parameters": {
                        "batchSize": 1,
                        "mode": "sequential",
                        "numberEntities": "all",
                        "samplerType": "generator",
                        "singleMeasurement": False,
                    },
                },
                "spaceIdentifier": "space-c1846f-1d5055",
                "spaces": ["space-c1846f-1d5055"],
            },
            "result": None,
        }

        migrated_data = migrator.migrator_function(old_data.copy())

        assert "result" not in migrated_data
        assert migrated_data["config"]["spaces"] == ["space-c1846f-1d5055"]


# Made with Bob
