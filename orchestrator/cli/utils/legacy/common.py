# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Common utilities for legacy validator handling"""

from typing import TYPE_CHECKING

import pydantic
from rich.console import Console

if TYPE_CHECKING:
    from orchestrator.core.legacy.metadata import LegacyValidatorMetadata
    from orchestrator.core.resources import CoreResourceKinds


def print_validator_suggestions(
    validators: list["LegacyValidatorMetadata"],
    resource_type: "CoreResourceKinds",
    console: Console,
    show_all_validators: bool = False,
) -> None:
    """Print legacy validator suggestions to the console

    Args:
        validators: List of applicable validators
        resource_type: The resource type
        console: Rich console to print to
        show_all_validators: If True, show all validators in the command example
    """
    # Resources can be referenced by their CoreResourceKinds value or by shorthands
    # from cli_shorthands_to_cli_names in orchestrator/cli/utils/resources/mappings.py
    resource_cli_name = resource_type.value

    console.print("\n[bold cyan]Available legacy validators:[/bold cyan]\n")

    for validator in validators:
        console.print(f"  • [green]{validator.identifier}[/green]")
        console.print(f"    {validator.description}")
        console.print(
            f"    Handles: {', '.join(validator.fully_qualified_deprecated_field_paths)}"
        )
        console.print(f"    Deprecated: v{validator.deprecated_from_version}")
        console.print()

    console.print("[bold magenta]To upgrade using legacy validators:[/bold magenta]")
    if show_all_validators:
        validator_args = " ".join(
            f"--apply-legacy-validator {v.identifier}" for v in validators
        )
    else:
        validator_args = f"--apply-legacy-validator {validators[0].identifier}"
    console.print(f"  ado upgrade {resource_cli_name} {validator_args}")
    console.print()
    console.print("[bold magenta]To list all legacy validators:[/bold magenta]")
    console.print(f"  ado upgrade {resource_cli_name} --list-legacy-validators")


def print_validator_suggestions_with_dependencies(
    validators: list["LegacyValidatorMetadata"],
    resource_type: "CoreResourceKinds",
    console: Console,
) -> None:
    """Print legacy validator suggestions with dependency information

    This enhanced version resolves dependencies and shows validators in the
    correct execution order, along with dependency information.

    Args:
        validators: List of applicable validators
        resource_type: The resource type
        console: Rich console to print to
    """
    from orchestrator.core.legacy.registry import LegacyValidatorRegistry

    # Resources can be referenced by their CoreResourceKinds value or by shorthands
    resource_cli_name = resource_type.value

    # Get validator identifiers
    validator_ids = [v.identifier for v in validators]

    # Resolve dependencies to get correct order
    try:
        ordered_ids, missing_deps = LegacyValidatorRegistry.resolve_dependencies(
            validator_ids
        )
    except ValueError as e:
        # Circular dependency detected
        console.print(
            f"\n[bold red]Warning:[/bold red] {e}",
            style="red",
        )
        # Fall back to original order
        ordered_ids = validator_ids
        missing_deps = []

    # Get ordered validators (filter out None values)
    ordered_validators: list[LegacyValidatorMetadata] = []
    for vid in ordered_ids:
        validator = LegacyValidatorRegistry.get_validator(vid)
        if validator is not None:
            ordered_validators.append(validator)

    console.print("\n[bold cyan]Available legacy validators:[/bold cyan]\n")

    for i, validator in enumerate(ordered_validators, 1):
        # Show execution order
        console.print(f"  {i}. [green]{validator.identifier}[/green]")
        console.print(f"     {validator.description}")
        console.print(
            f"     Handles: {', '.join(validator.fully_qualified_deprecated_field_paths)}"
        )
        console.print(f"     Deprecated: v{validator.deprecated_from_version}")

        # Show dependencies if any
        if validator.dependencies:
            dep_names = []
            for dep_id in validator.dependencies:
                dep_validator = LegacyValidatorRegistry.get_validator(dep_id)
                if dep_validator:
                    dep_names.append(dep_validator.identifier)
                else:
                    dep_names.append(f"{dep_id} [red](missing)[/red]")
            console.print(f"     Dependencies: {', '.join(dep_names)}")

        console.print()

    # Warn about missing dependencies
    if missing_deps:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Some dependencies are missing: {', '.join(missing_deps)}\n"
        )

    console.print("[bold magenta]To upgrade using legacy validators:[/bold magenta]")

    # Build command with all validators in correct order
    validator_args = " ".join(
        f"--apply-legacy-validator {v.identifier}" for v in ordered_validators
    )
    console.print(f"  ado upgrade {resource_cli_name} {validator_args}")
    console.print()

    # Show note about automatic dependency resolution
    if len(ordered_validators) > len(validators):
        console.print(
            "[dim]Note: Additional validators were included to satisfy dependencies[/dim]\n"
        )

    console.print("[bold magenta]To list all legacy validators:[/bold magenta]")
    console.print(f"  ado upgrade {resource_cli_name} --list-legacy-validators")


# Made with Bob


def extract_deprecated_fields_from_validation_error(
    error: pydantic.ValidationError,
) -> tuple[set[str], dict[str, list[str]]]:
    """Extract field paths and error details from pydantic validation errors

    Args:
        error: The pydantic validation error

    Returns:
        Tuple of (full field paths, field error details mapping)
        - full field paths: Set of full dotted paths like 'config.specification.module.moduleType'
        - field error details: Maps full field path to list of error messages
    """
    fully_qualified_deprecated_field_paths: set[str] = set()
    field_errors: dict[str, list[str]] = {}

    for err in error.errors():
        if err.get("loc"):
            # Build the full dotted path from the location tuple
            full_path = ".".join(str(loc) for loc in err["loc"])
            fully_qualified_deprecated_field_paths.add(full_path)

            # Store the error message for this field path
            if full_path not in field_errors:
                field_errors[full_path] = []

            # Build a descriptive error message
            msg = err.get("msg", "")
            if err.get("input"):
                msg = f"{msg} (got: {err['input']})"

            field_errors[full_path].append(msg)

    return fully_qualified_deprecated_field_paths, field_errors


def extract_deprecated_fields_from_value_error(
    error: ValueError,
    resource_type: "CoreResourceKinds",
) -> tuple[set[str], dict[str, list[str]]]:
    """Extract field paths from ValueError containing pydantic validation errors

    This function attempts to extract the underlying pydantic ValidationError
    from a ValueError and extract field paths from it. If that fails, it falls
    back to simple string matching on the error message using known field paths
    from the legacy validator registry.

    Args:
        error: The ValueError that may contain a pydantic ValidationError
        resource_type: The resource type to get field paths for

    Returns:
        Tuple of (full field paths, field error details mapping)
    """
    # Try to extract pydantic ValidationError from the ValueError
    if hasattr(error, "__cause__") and isinstance(
        error.__cause__, pydantic.ValidationError
    ):
        return extract_deprecated_fields_from_validation_error(error.__cause__)

    # Fallback to simple string matching on error message
    from orchestrator.core.legacy.registry import LegacyValidatorRegistry

    error_msg = str(error)
    fully_qualified_deprecated_field_paths: set[str] = set()
    field_errors: dict[str, list[str]] = {}

    # Get all field paths from registered validators for this resource type
    validators = LegacyValidatorRegistry.get_validators_for_resource(resource_type)
    known_fully_qualified_deprecated_field_paths = {
        path
        for validator in validators
        for path in validator.fully_qualified_deprecated_field_paths
    }

    for field_path in known_fully_qualified_deprecated_field_paths:
        if field_path in error_msg:
            fully_qualified_deprecated_field_paths.add(field_path)
            # For string matching fallback, we don't have detailed error messages
            field_errors[field_path] = [
                "Field validation failed (details in error message)"
            ]

    return fully_qualified_deprecated_field_paths, field_errors
