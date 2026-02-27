# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for legacy validators with pydantic models and upgrade process"""

from unittest.mock import MagicMock, patch

import pydantic
import pytest

from orchestrator.core.legacy.registry import LegacyValidatorRegistry, legacy_validator
from orchestrator.core.resources import CoreResourceKinds


class TestLegacyValidatorWithPydantic:
    """Test legacy validators working with pydantic models"""

    def setup_method(self) -> None:
        """Clear the registry before each test"""
        LegacyValidatorRegistry._validators = {}

    def test_validator_applied_during_model_validation(self) -> None:
        """Test that a legacy validator can be manually applied before pydantic validation"""

        # Define a simple pydantic model
        class OldModel(pydantic.BaseModel):
            new_field: str

        # Register a legacy validator
        @legacy_validator(
            identifier="old_to_new_field",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["old_field"],
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

    def test_csv_sample_store_migration_validator(self) -> None:
        """Test the CSV sample store migration validator with realistic data"""

        # Import the validator to register it
        from orchestrator.core.legacy.validators.samplestore.v1_to_v2_csv_migration import (  # noqa: F401
            migrate_csv_v1_to_v2,
        )

        # Get the validator
        validator = LegacyValidatorRegistry.get_validator(
            "csv_constitutive_columns_migration"
        )
        assert validator is not None
        assert validator.resource_type == CoreResourceKinds.SAMPLESTORE

        # Old format CSV sample store data
        old_csv_data = {
            "kind": "samplestore",
            "type": "csv",
            "identifier": "test_store",
            "identifierColumn": "id",
            "constitutivePropertyColumns": ["prop1", "prop2"],
            "experiments": [
                {
                    "experimentIdentifier": "exp1",
                    "actuatorIdentifier": "act1",
                    "propertyMap": ["obs1", "obs2"],
                }
            ],
        }

        # Apply the validator
        migrated_data = validator.validator_function(old_csv_data.copy())

        # Verify migration
        assert "constitutivePropertyColumns" not in migrated_data
        assert len(migrated_data["experiments"]) == 1
        exp = migrated_data["experiments"][0]
        assert "propertyMap" not in exp
        assert "observedPropertyMap" in exp
        assert exp["observedPropertyMap"] == ["obs1", "obs2"]
        assert "constitutivePropertyMap" in exp
        assert exp["constitutivePropertyMap"] == ["prop1", "prop2"]

    def test_entitysource_to_samplestore_migration(self) -> None:
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

    def test_chained_validators(self) -> None:
        """Test applying multiple validators in sequence"""

        # Register two validators
        @legacy_validator(
            identifier="step1_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["old_field1"],
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
            deprecated_fields=["intermediate_field"],
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
    """Test the upgrade handler with legacy validators"""

    def setup_method(self) -> None:
        """Clear the registry before each test"""
        LegacyValidatorRegistry._validators = {}

    def test_upgrade_handler_applies_legacy_validator(self) -> None:
        """Test that handle_ado_upgrade applies legacy validators correctly"""

        # Register a test validator
        @legacy_validator(
            identifier="test_upgrade_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test upgrade validator",
        )
        def test_validator(data: dict) -> dict:
            if "old_field" in data:
                data["new_field"] = data.pop("old_field")
            return data

        # Create a mock resource class with model_validate
        mock_resource_class = MagicMock()
        mock_validated_resource = MagicMock()
        mock_resource_class.model_validate.return_value = mock_validated_resource

        # Create mock resource instance
        mock_resource = MagicMock()
        mock_resource.model_dump.return_value = {"old_field": "test_value"}
        # Configure type() to return our mock class
        type(mock_resource).model_validate = mock_resource_class.model_validate

        mock_sql_store = MagicMock()
        mock_sql_store.getResourcesOfKind.return_value = {"res1": mock_resource}

        # Mock parameters
        mock_params = MagicMock()
        mock_params.apply_legacy_validator = ["test_upgrade_validator"]
        mock_params.list_legacy_validators = False
        mock_params.ado_configuration.project_context = "test_context"

        # Patch dependencies
        with (
            patch(
                "orchestrator.cli.utils.resources.handlers.get_sql_store",
                return_value=mock_sql_store,
            ),
            patch(
                "orchestrator.cli.utils.resources.handlers._import_legacy_validators"
            ),
            patch("orchestrator.cli.utils.resources.handlers.Status"),
            patch("orchestrator.cli.utils.resources.handlers.console_print"),
        ):
            from orchestrator.cli.utils.resources.handlers import (
                handle_ado_upgrade,
            )

            # Call the upgrade handler
            handle_ado_upgrade(
                parameters=mock_params,
                resource_type=CoreResourceKinds.SAMPLESTORE,
            )

        # Verify the resource was processed
        mock_resource.model_dump.assert_called_once()
        mock_sql_store.updateResource.assert_called_once()

    def test_upgrade_handler_validates_validator_resource_type(self) -> None:
        """Test that upgrade handler validates validator resource type matches"""

        # Register a validator for OPERATION
        @legacy_validator(
            identifier="operation_validator",
            resource_type=CoreResourceKinds.OPERATION,
            deprecated_fields=["old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Operation validator",
        )
        def op_validator(data: dict) -> dict:
            return data

        # Mock parameters trying to use operation validator on samplestore
        mock_params = MagicMock()
        mock_params.apply_legacy_validator = ["operation_validator"]
        mock_params.list_legacy_validators = False
        mock_params.ado_configuration.project_context = "test_context"

        mock_sql_store = MagicMock()

        # Patch dependencies
        with (
            patch(
                "orchestrator.cli.utils.resources.handlers.get_sql_store",
                return_value=mock_sql_store,
            ),
            patch(
                "orchestrator.cli.utils.resources.handlers._import_legacy_validators"
            ),
            patch(
                "orchestrator.cli.utils.resources.handlers.console_print"
            ) as mock_print,
        ):
            import typer

            from orchestrator.cli.utils.resources.handlers import (
                handle_ado_upgrade,
            )

            # Should raise typer.Exit
            with pytest.raises(typer.Exit) as exc_info:
                handle_ado_upgrade(
                    parameters=mock_params,
                    resource_type=CoreResourceKinds.SAMPLESTORE,
                )

            assert exc_info.value.exit_code == 1

            # Verify error message was printed
            mock_print.assert_called()
            call_args = str(mock_print.call_args)
            assert "operation_validator" in call_args
            assert "operation" in call_args.lower()
            assert "samplestore" in call_args.lower()

    def test_upgrade_handler_validates_validator_exists(self) -> None:
        """Test that upgrade handler validates validator exists"""

        # Mock parameters with non-existent validator
        mock_params = MagicMock()
        mock_params.apply_legacy_validator = ["nonexistent_validator"]
        mock_params.list_legacy_validators = False
        mock_params.ado_configuration.project_context = "test_context"

        mock_sql_store = MagicMock()

        # Patch dependencies
        with (
            patch(
                "orchestrator.cli.utils.resources.handlers.get_sql_store",
                return_value=mock_sql_store,
            ),
            patch(
                "orchestrator.cli.utils.resources.handlers._import_legacy_validators"
            ),
            patch(
                "orchestrator.cli.utils.resources.handlers.console_print"
            ) as mock_print,
        ):
            import typer

            from orchestrator.cli.utils.resources.handlers import (
                handle_ado_upgrade,
            )

            # Should raise typer.Exit
            with pytest.raises(typer.Exit) as exc_info:
                handle_ado_upgrade(
                    parameters=mock_params,
                    resource_type=CoreResourceKinds.SAMPLESTORE,
                )

            assert exc_info.value.exit_code == 1

            # Verify error message was printed
            mock_print.assert_called()
            call_args = str(mock_print.call_args)
            assert "nonexistent_validator" in call_args
            assert "not found" in call_args.lower()


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
            deprecated_fields=["old_field"],
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

    def test_validator_handles_missing_fields_gracefully(self) -> None:
        """Test that validators handle missing deprecated fields gracefully"""

        @legacy_validator(
            identifier="graceful_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["optional_old_field"],
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
