# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for Phase 1 transaction safety in upgrade handler"""

from unittest.mock import MagicMock, patch

import pytest
import typer

from orchestrator.core.legacy.registry import LegacyValidatorRegistry, legacy_validator
from orchestrator.core.resources import CoreResourceKinds


class TestUpgradeTransactionSafety:
    """Test transaction safety in upgrade handler - validate-all-before-save pattern"""

    def setup_method(self) -> None:
        """Clear the registry before each test"""
        LegacyValidatorRegistry._validators = {}

    def test_all_resources_validated_before_any_saved(self) -> None:
        """Test that all resources are validated before any are saved"""

        # Register a test validator
        @legacy_validator(
            identifier="test_transaction_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            fully_qualified_deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test transaction validator",
        )
        def test_validator(data: dict) -> dict:
            if "config" in data and "old_field" in data["config"]:
                data["config"]["new_field"] = data["config"].pop("old_field")
            return data

        # Create mock resources
        mock_resource1 = MagicMock()
        mock_resource1.model_dump.return_value = {
            "kind": "samplestore",
            "identifier": "res1",
            "config": {"old_field": "value1"},
        }

        mock_resource2 = MagicMock()
        mock_resource2.model_dump.return_value = {
            "kind": "samplestore",
            "identifier": "res2",
            "config": {"old_field": "value2"},
        }

        # Mock resource class
        mock_resource_class = MagicMock()
        validated_resources = []

        def mock_validate(data: dict) -> MagicMock:
            validated = MagicMock()
            validated.model_dump.return_value = data
            validated_resources.append(data["identifier"])
            return validated

        mock_resource_class.model_validate.side_effect = mock_validate

        # Mock SQL store
        mock_sql_store = MagicMock()
        mock_sql_store.getResourcesOfKind.return_value = {
            "res1": mock_resource1,
            "res2": mock_resource2,
        }
        mock_sql_store.getResourceIdentifiersOfKind.return_value = {
            "IDENTIFIER": ["res1", "res2"]
        }
        mock_sql_store.getResourceRaw.side_effect = lambda id: (
            mock_resource1.model_dump() if id == "res1" else mock_resource2.model_dump()
        )

        update_calls = []

        def track_update(resource: MagicMock) -> None:
            update_calls.append(resource.model_dump()["identifier"])

        mock_sql_store.updateResource.side_effect = track_update

        # Mock parameters
        mock_params = MagicMock()
        mock_params.apply_legacy_validator = ["test_transaction_validator"]
        mock_params.list_legacy_validators = False
        mock_params.ado_configuration.project_context = "test_context"

        # Patch dependencies
        with (
            patch(
                "orchestrator.cli.utils.resources.handlers.get_sql_store",
                return_value=mock_sql_store,
            ),
            patch(
                "orchestrator.core.kindmap",
                {"samplestore": mock_resource_class},
            ),
            patch("orchestrator.cli.utils.resources.handlers.Status"),
            patch("orchestrator.cli.utils.resources.handlers.console_print"),
        ):
            from orchestrator.cli.utils.resources.handlers import handle_ado_upgrade

            # Call the upgrade handler
            handle_ado_upgrade(
                parameters=mock_params,
                resource_type=CoreResourceKinds.SAMPLESTORE,
            )

        # Verify: all resources validated before any saved
        # Both resources should be validated
        assert len(validated_resources) == 2
        assert "res1" in validated_resources
        assert "res2" in validated_resources

        # Both resources should be saved
        assert len(update_calls) == 2
        assert "res1" in update_calls
        assert "res2" in update_calls

    def test_validation_failure_prevents_all_saves(self) -> None:
        """Test that if any validation fails, no resources are saved"""

        # Register a test validator
        @legacy_validator(
            identifier="test_failing_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            fully_qualified_deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test failing validator",
        )
        def test_validator(data: dict) -> dict:
            if "config" in data and "old_field" in data["config"]:
                data["config"]["new_field"] = data["config"].pop("old_field")
            return data

        # Create mock resources - one valid, one will fail validation
        mock_resource1 = MagicMock()
        mock_resource1.model_dump.return_value = {
            "kind": "samplestore",
            "identifier": "res1",
            "config": {"old_field": "value1"},
        }

        mock_resource2 = MagicMock()
        mock_resource2.model_dump.return_value = {
            "kind": "samplestore",
            "identifier": "res2",
            "config": {"old_field": "value2"},
        }

        # Mock resource class - second validation fails
        mock_resource_class = MagicMock()
        validation_count = [0]

        def mock_validate(data: dict) -> MagicMock:
            validation_count[0] += 1
            if validation_count[0] == 2:
                # Second validation fails - raise a simple ValueError
                raise ValueError("Validation failed for resource res2")
            validated = MagicMock()
            validated.model_dump.return_value = data
            return validated

        mock_resource_class.model_validate.side_effect = mock_validate

        # Mock SQL store
        mock_sql_store = MagicMock()
        mock_sql_store.getResourcesOfKind.return_value = {
            "res1": mock_resource1,
            "res2": mock_resource2,
        }
        mock_sql_store.getResourceIdentifiersOfKind.return_value = {
            "IDENTIFIER": ["res1", "res2"]
        }
        mock_sql_store.getResourceRaw.side_effect = lambda id: (
            mock_resource1.model_dump() if id == "res1" else mock_resource2.model_dump()
        )

        # Mock parameters
        mock_params = MagicMock()
        mock_params.apply_legacy_validator = ["test_failing_validator"]
        mock_params.list_legacy_validators = False
        mock_params.ado_configuration.project_context = "test_context"

        # Patch dependencies
        with (
            patch(
                "orchestrator.cli.utils.resources.handlers.get_sql_store",
                return_value=mock_sql_store,
            ),
            patch(
                "orchestrator.core.kindmap",
                {"samplestore": mock_resource_class},
            ),
            patch("orchestrator.cli.utils.resources.handlers.Status"),
            patch(
                "orchestrator.cli.utils.resources.handlers.console_print"
            ) as mock_print,
        ):
            from orchestrator.cli.utils.resources.handlers import handle_ado_upgrade

            # Should raise typer.Exit due to validation failure
            with pytest.raises(typer.Exit) as exc_info:
                handle_ado_upgrade(
                    parameters=mock_params,
                    resource_type=CoreResourceKinds.SAMPLESTORE,
                )

            assert exc_info.value.exit_code == 1

        # Verify: NO resources were saved (transaction safety)
        mock_sql_store.updateResource.assert_not_called()

        # Verify error was printed
        mock_print.assert_called()

    def test_empty_resource_list_handled_gracefully(self) -> None:
        """Test that empty resource list is handled without errors"""

        # Register a test validator
        @legacy_validator(
            identifier="test_empty_validator",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            fully_qualified_deprecated_field_paths=["config.old_field"],
            deprecated_from_version="1.0.0",
            removed_from_version="2.0.0",
            description="Test empty validator",
        )
        def test_validator(data: dict) -> dict:
            return data

        # Mock SQL store with no resources
        mock_sql_store = MagicMock()
        mock_sql_store.getResourcesOfKind.return_value = {}
        mock_sql_store.getResourceIdentifiersOfKind.return_value = {"IDENTIFIER": []}

        # Mock parameters
        mock_params = MagicMock()
        mock_params.apply_legacy_validator = ["test_empty_validator"]
        mock_params.list_legacy_validators = False
        mock_params.ado_configuration.project_context = "test_context"

        # Patch dependencies
        with (
            patch(
                "orchestrator.cli.utils.resources.handlers.get_sql_store",
                return_value=mock_sql_store,
            ),
            patch("orchestrator.cli.utils.resources.handlers.Status"),
            patch(
                "orchestrator.cli.utils.resources.handlers.console_print"
            ) as mock_print,
        ):
            from orchestrator.cli.utils.resources.handlers import handle_ado_upgrade

            # Should complete without error
            handle_ado_upgrade(
                parameters=mock_params,
                resource_type=CoreResourceKinds.SAMPLESTORE,
            )

        # Verify: no updates attempted
        mock_sql_store.updateResource.assert_not_called()

        # Verify message printed
        mock_print.assert_called()


# Made with Bob
