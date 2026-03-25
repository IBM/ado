# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for Phase 1 transaction safety in upgrade handler"""

import json

import pytest
import sqlalchemy
import typer

from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.cli.models.parameters import AdoUpgradeCommandParameters
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.resources.handlers import handle_ado_upgrade
from orchestrator.core.legacy.registry import LegacyValidatorRegistry, legacy_validator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)
from orchestrator.core.samplestore.resource import SampleStoreResource
from orchestrator.metastore.project import ProjectContext


class TestUpgradeTransactionSafety:
    """Test transaction safety in upgrade handler - validate-all-before-save pattern"""

    def setup_method(self) -> None:
        """Clear the registry before each test"""
        LegacyValidatorRegistry._validators = {}

    @pytest.mark.parametrize("valid_ado_project_context", ["sqlite"], indirect=True)
    def test_all_resources_validated_before_any_saved(
        self,
        valid_ado_project_context: ProjectContext,
    ) -> None:
        """Test that all resources are validated before any are saved"""

        # Register a test validator that transforms old_field -> new_field
        @legacy_validator(
            identifier="test_transaction_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.metadata.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test transaction validator",
        )
        def test_validator(data: dict) -> dict:
            if "config" in data and "metadata" in data["config"]:
                metadata = data["config"]["metadata"]
                if "old_field" in metadata:
                    metadata["new_field"] = metadata.pop("old_field")
            return data

        # Create two sample store resources with old_field in metadata
        resource1 = SampleStoreResource(
            identifier="test_res1",
            config=SampleStoreConfiguration(
                specification=SampleStoreSpecification(
                    module=SampleStoreModuleConf(
                        moduleClass="SQLSampleStore",
                        moduleName="orchestrator.core.samplestore.sql",
                    ),
                    storageLocation=valid_ado_project_context.metadataStore,
                ),
                metadata={"old_field": "value1"},
            ),
        )

        resource2 = SampleStoreResource(
            identifier="test_res2",
            config=SampleStoreConfiguration(
                specification=SampleStoreSpecification(
                    module=SampleStoreModuleConf(
                        moduleClass="SQLSampleStore",
                        moduleName="orchestrator.core.samplestore.sql",
                    ),
                    storageLocation=valid_ado_project_context.metadataStore,
                ),
                metadata={"old_field": "value2"},
            ),
        )

        # Save resources to database
        sql_store = get_sql_store(project_context=valid_ado_project_context)
        sql_store.updateResource(resource=resource1)
        sql_store.updateResource(resource=resource2)

        # Now manually add the deprecated field to the raw data in the database
        # We need to update the JSON directly in the database
        with sql_store.engine.begin() as conn:
            # Get current data
            raw1 = sql_store.getResourceRaw("test_res1")
            raw1["config"]["metadata"]["old_field"] = "value1"

            # Update in database
            update_stmt = sqlalchemy.text(
                "UPDATE resources SET data = :data WHERE identifier = :identifier"
            ).bindparams(data=json.dumps(raw1), identifier="test_res1")
            conn.execute(update_stmt)

            # Same for resource2
            raw2 = sql_store.getResourceRaw("test_res2")
            raw2["config"]["metadata"]["old_field"] = "value2"

            update_stmt = sqlalchemy.text(
                "UPDATE resources SET data = :data WHERE identifier = :identifier"
            ).bindparams(data=json.dumps(raw2), identifier="test_res2")
            conn.execute(update_stmt)

        # Create parameters for upgrade
        ado_config = AdoConfiguration()
        ado_config._project_context = valid_ado_project_context
        params = AdoUpgradeCommandParameters(
            ado_configuration=ado_config,
            apply_legacy_validator=["test_transaction_validator"],
            list_legacy_validators=False,
        )

        # Call the upgrade handler
        handle_ado_upgrade(
            parameters=params,
            resource_type=CoreResourceKinds.SAMPLESTORE,
        )

        # Verify both resources were upgraded
        upgraded_res1 = sql_store.getResourceRaw("test_res1")
        upgraded_res2 = sql_store.getResourceRaw("test_res2")

        assert upgraded_res1 is not None
        assert upgraded_res2 is not None
        assert "new_field" in upgraded_res1["config"]["metadata"]
        assert "new_field" in upgraded_res2["config"]["metadata"]
        assert upgraded_res1["config"]["metadata"]["new_field"] == "value1"
        assert upgraded_res2["config"]["metadata"]["new_field"] == "value2"
        assert "old_field" not in upgraded_res1["config"]["metadata"]
        assert "old_field" not in upgraded_res2["config"]["metadata"]

    @pytest.mark.parametrize("valid_ado_project_context", ["sqlite"], indirect=True)
    def test_validation_failure_prevents_all_saves(
        self,
        valid_ado_project_context: ProjectContext,
    ) -> None:
        """Test that if any validation fails, no resources are saved"""

        # Register a validator that will cause validation failure
        @legacy_validator(
            identifier="test_failing_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.metadata.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test failing validator",
        )
        def test_validator(data: dict) -> dict:
            # Transform the field
            if "config" in data and "metadata" in data["config"]:
                metadata = data["config"]["metadata"]
                if "old_field" in metadata:
                    metadata["new_field"] = metadata.pop("old_field")

            # Introduce an invalid field that will fail pydantic validation
            # for the second resource only
            if data.get("identifier") == "test_res2":
                data["config"]["invalid_field_that_breaks_validation"] = "bad_value"

            return data

        # Create two sample store resources
        resource1 = SampleStoreResource(
            identifier="test_res1",
            config=SampleStoreConfiguration(
                specification=SampleStoreSpecification(
                    module=SampleStoreModuleConf(
                        moduleClass="SQLSampleStore",
                        moduleName="orchestrator.core.samplestore.sql",
                    ),
                    storageLocation=valid_ado_project_context.metadataStore,
                ),
                metadata={"old_field": "value1"},
            ),
        )

        resource2 = SampleStoreResource(
            identifier="test_res2",
            config=SampleStoreConfiguration(
                specification=SampleStoreSpecification(
                    module=SampleStoreModuleConf(
                        moduleClass="SQLSampleStore",
                        moduleName="orchestrator.core.samplestore.sql",
                    ),
                    storageLocation=valid_ado_project_context.metadataStore,
                ),
                metadata={"old_field": "value2"},
            ),
        )

        # Save resources to database
        sql_store = get_sql_store(project_context=valid_ado_project_context)
        sql_store.updateResource(resource=resource1)
        sql_store.updateResource(resource=resource2)

        # Now manually add the deprecated field to the raw data in the database
        with sql_store.engine.begin() as conn:
            # Get current data
            raw1 = sql_store.getResourceRaw("test_res1")
            raw1["config"]["metadata"]["old_field"] = "value1"

            # Update in database
            update_stmt = sqlalchemy.text(
                "UPDATE resources SET data = :data WHERE identifier = :identifier"
            ).bindparams(data=json.dumps(raw1), identifier="test_res1")
            conn.execute(update_stmt)

            # Same for resource2
            raw2 = sql_store.getResourceRaw("test_res2")
            raw2["config"]["metadata"]["old_field"] = "value2"

            update_stmt = sqlalchemy.text(
                "UPDATE resources SET data = :data WHERE identifier = :identifier"
            ).bindparams(data=json.dumps(raw2), identifier="test_res2")
            conn.execute(update_stmt)

        # Store original data for comparison
        original_res1 = sql_store.getResourceRaw("test_res1")
        original_res2 = sql_store.getResourceRaw("test_res2")

        # Create parameters for upgrade
        ado_config = AdoConfiguration()
        ado_config._project_context = valid_ado_project_context
        params = AdoUpgradeCommandParameters(
            ado_configuration=ado_config,
            apply_legacy_validator=["test_failing_validator"],
            list_legacy_validators=False,
        )

        # Should raise typer.Exit due to validation failure
        with pytest.raises(typer.Exit) as exc_info:
            handle_ado_upgrade(
                parameters=params,
                resource_type=CoreResourceKinds.SAMPLESTORE,
            )

        assert exc_info.value.exit_code == 1

        # Verify: NO resources were saved (transaction safety)
        # Both resources should still have their original data
        current_res1 = sql_store.getResourceRaw("test_res1")
        current_res2 = sql_store.getResourceRaw("test_res2")

        assert current_res1 == original_res1
        assert current_res2 == original_res2
        assert "old_field" in current_res1["config"]["metadata"]
        assert "old_field" in current_res2["config"]["metadata"]
        assert "new_field" not in current_res1["config"]["metadata"]
        assert "new_field" not in current_res2["config"]["metadata"]

    def test_empty_resource_list_handled_gracefully(
        self,
        valid_ado_project_context: ProjectContext,
    ) -> None:
        """Test that empty resource list is handled without errors"""

        # Register a test validator
        @legacy_validator(
            identifier="test_empty_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.metadata.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test empty validator",
        )
        def test_validator(data: dict) -> dict:
            return data

        # Don't create any resources - database starts empty for this test

        # Create parameters for upgrade
        ado_config = AdoConfiguration()
        ado_config._project_context = valid_ado_project_context
        params = AdoUpgradeCommandParameters(
            ado_configuration=ado_config,
            apply_legacy_validator=["test_empty_validator"],
            list_legacy_validators=False,
        )

        # Should complete without error
        handle_ado_upgrade(
            parameters=params,
            resource_type=CoreResourceKinds.SAMPLESTORE,
        )

        # Verify no samplestore resources exist
        sql_store = get_sql_store(project_context=valid_ado_project_context)
        resources = sql_store.getResourcesOfKind(
            kind=CoreResourceKinds.SAMPLESTORE.value
        )
        assert len(resources) == 0


# Made with Bob
