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


def extract_deprecated_fields_from_validation_error(
    error: pydantic.ValidationError,
) -> list[str]:
    """Extract field names from pydantic validation errors

    Args:
        error: The pydantic validation error

    Returns:
        List of field names that caused validation errors
    """
    deprecated_fields = []
    for err in error.errors():
        # Get the field path from the error
        if err.get("loc"):
            # loc is a tuple of field names in the path
            field_name = str(err["loc"][0])
            if field_name not in deprecated_fields:
                deprecated_fields.append(field_name)
    return deprecated_fields


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
    from orchestrator.core.legacy.registry import LegacyValidatorRegistry

    # Import validators to ensure they're registered
    _import_legacy_validators()

    # Extract field names from validation error
    deprecated_fields = extract_deprecated_fields_from_validation_error(error)

    if not deprecated_fields:
        # No fields extracted, show standard error
        console_print(f"Validation error: {error}", stderr=True)
        raise typer.Exit(1) from error

    # Find applicable legacy validators
    validators = LegacyValidatorRegistry.find_validators_for_fields(
        resource_type=resource_type, field_names=deprecated_fields
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
        f"\nDeprecated fields detected: [yellow]{', '.join(deprecated_fields)}[/yellow]"
    )
    console.print("\n[bold cyan]Available legacy validators:[/bold cyan]")

    # Map resource types to their CLI names
    resource_name_mapping = {
        CoreResourceKinds.SAMPLESTORE: "sample_stores",
        CoreResourceKinds.DISCOVERYSPACE: "spaces",
        CoreResourceKinds.OPERATION: "operations",
        CoreResourceKinds.ACTUATORCONFIGURATION: "actuator_configurations",
        CoreResourceKinds.DATACONTAINER: "data_containers",
    }
    resource_cli_name = resource_name_mapping.get(
        resource_type, resource_type.value + "s"
    )

    for validator in validators:
        console.print(f"  • [green]{validator.identifier}[/green]")
        console.print(f"    {validator.description}")
        console.print(f"    Handles: {', '.join(validator.deprecated_fields)}")
        console.print(f"    Deprecated: v{validator.deprecated_from_version}")
        console.print()

    console.print("[bold magenta]To upgrade using a legacy validator:[/bold magenta]")
    console.print(
        f"  ado upgrade {resource_cli_name} --apply-legacy-validator {validators[0].identifier}"
    )
    console.print()
    console.print("[bold magenta]To list all legacy validators:[/bold magenta]")
    console.print(f"  ado upgrade {resource_cli_name} --list-legacy-validators")
    console.print()

    raise typer.Exit(1) from error


def _import_legacy_validators() -> None:
    """Import all legacy validator modules to ensure they're registered"""
    # Import validator modules to trigger decorator registration
    try:
        import orchestrator.core.legacy.validators.resource.entitysource_to_samplestore  # noqa: F401
        import orchestrator.core.legacy.validators.samplestore.v1_to_v2_csv_migration  # noqa: F401
    except ImportError:
        pass  # Validators may not be available in all installations
