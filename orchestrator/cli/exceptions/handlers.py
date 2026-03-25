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
    # Import validators package to trigger registration via __init__.py
    import orchestrator.core.legacy.validators  # noqa: F401
    from orchestrator.cli.utils.legacy.common import (
        extract_deprecated_field_paths_from_validation_error,
        print_validator_suggestions_with_dependencies,
    )
    from orchestrator.core.legacy.registry import LegacyValidatorRegistry

    # Extract field paths and error details from validation error
    deprecated_field_paths, field_errors = (
        extract_deprecated_field_paths_from_validation_error(error)
    )
    if not deprecated_field_paths:
        # No fields extracted, show standard error
        console_print(f"Validation error: {error}", stderr=True)
        raise typer.Exit(1) from error

    # Find applicable legacy validators using full field paths for precise matching
    validators = LegacyValidatorRegistry.find_validators_for_deprecated_field_paths(
        resource_type=resource_type,
        deprecated_field_paths=deprecated_field_paths,
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
        f"\n[bold]Fields with validation errors:[/bold] [yellow]{len(deprecated_field_paths)} field(s)[/yellow]"
    )
    # Show detailed error messages for each field path
    console.print("\n[bold]Error details:[/bold]")
    for field_path in sorted(deprecated_field_paths):
        console.print(f"  • [cyan]{field_path}[/cyan]:")
        for error_msg in field_errors.get(field_path, []):
            console.print(f"    - {error_msg}")
    console.print()

    # Use enhanced suggestion printer with dependency information
    print_validator_suggestions_with_dependencies(
        validators=validators,
        resource_type=resource_type,
        console=console,
    )

    raise typer.Exit(1) from error
