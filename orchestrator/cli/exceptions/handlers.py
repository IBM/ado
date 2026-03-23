# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import NoReturn

import pydantic
import typer
from rich.console import Console

from orchestrator.cli.utils.output.prints import (
    console_print,
    could_not_delete_resource_from_database_error_str,
    no_related_resources_error_str,
    no_resource_with_id_in_db_error_str,
    unknown_experiment_error_str,
)
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.base import (
    DeleteFromDatabaseError,
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.actuators.registry import UnknownExperimentError


def handle_resource_does_not_exist(
    error: ResourceDoesNotExistError, project_context: ProjectContext
) -> NoReturn:
    console_print(
        no_resource_with_id_in_db_error_str(
            resource_id=error.resource_id,
            kind=error.kind,
            context=project_context.project,
        ),
        stderr=True,
    )
    raise typer.Exit(1) from error


def handle_no_related_resource(
    error: NoRelatedResourcesError, project_context: ProjectContext
) -> NoReturn:
    console_print(
        no_related_resources_error_str(
            resource_id=error.resource_id,
            kind=error.kind,
            context=project_context.project,
        ),
        stderr=True,
    )
    raise typer.Exit(1) from error


def handle_unknown_experiment_error(error: UnknownExperimentError) -> NoReturn:
    console_print(unknown_experiment_error_str(error=error), stderr=True)
    raise typer.Exit(1) from error


def handle_resource_deletion_error(error: DeleteFromDatabaseError) -> NoReturn:
    console_print(
        could_not_delete_resource_from_database_error_str(
            error=error,
        ),
        stderr=True,
    )
    raise typer.Exit(1) from error


def handle_validation_error_with_legacy_suggestions(
    error: pydantic.ValidationError,
    resource_type: CoreResourceKinds,
    resource_identifier: str | None = None,
) -> NoReturn:
    """Handle pydantic validation errors and suggest legacy validators if applicable

    Args:
        error: The pydantic validation error
        resource_type: The type of resource that failed validation
        resource_identifier: Optional identifier of the resource

    Raises:
        typer.Exit: Always exits with code 1
    """
    from orchestrator.cli.utils.legacy.common import (
        extract_deprecated_fields_from_validation_error,
        import_legacy_validators,
        print_validator_suggestions,
    )
    from orchestrator.core.legacy.registry import LegacyValidatorRegistry

    # Import validators to ensure they're registered
    import_legacy_validators()

    # Extract field paths, error details, and leaf field names from validation error
    full_field_paths, field_errors, leaf_field_names = (
        extract_deprecated_fields_from_validation_error(error)
    )
    if not full_field_paths:
        # No fields extracted, show standard error
        console_print(f"Validation error: {error}", stderr=True)
        raise typer.Exit(1) from error

    # Find applicable legacy validators using leaf field names for better matching
    validators = LegacyValidatorRegistry.find_validators_for_fields(
        resource_type=resource_type, field_names=leaf_field_names
    )

    if not validators:
        # No legacy validators available, show standard error
        console_print(f"Validation error: {error}", stderr=True)
        raise typer.Exit(1) from error

    # Display helpful error message with suggestions
    console = Console(stderr=True)
    resource_id_str = f" '{resource_identifier}'" if resource_identifier else ""
    console.print(
        f"\n[bold red]Validation Error[/bold red] in {resource_type.value}{resource_id_str}"
    )
    console.print(
        f"\n[bold]Fields with validation errors:[/bold] [yellow]{len(full_field_paths)} field(s)[/yellow]"
    )
    # Show detailed error messages for each field path
    console.print("\n[bold]Error details:[/bold]")
    for field_path in sorted(full_field_paths):
        console.print(f"  • [cyan]{field_path}[/cyan]:")
        for error_msg in field_errors.get(field_path, []):
            console.print(f"    - {error_msg}")
    console.print()

    print_validator_suggestions(
        validators=validators,
        resource_type=resource_type,
        console=console,
        show_all_validators=False,
    )

    raise typer.Exit(1) from error
