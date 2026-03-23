# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Common utilities for legacy validator handling"""

from typing import TYPE_CHECKING

import pydantic
from rich.console import Console

if TYPE_CHECKING:
    from orchestrator.core.legacy.metadata import LegacyValidatorMetadata
    from orchestrator.core.resources import CoreResourceKinds


def import_legacy_validators() -> None:
    """Import all legacy validator modules to ensure they're registered"""
    # Import validator modules to trigger decorator registration
    try:
        # Discovery Space validators
        import orchestrator.core.legacy.validators.discoveryspace.entitysource_to_samplestore  # noqa: F401
        import orchestrator.core.legacy.validators.discoveryspace.properties_field_removal  # noqa: F401

        # Operation validators
        import orchestrator.core.legacy.validators.operation.actuators_field_removal  # noqa: F401
        import orchestrator.core.legacy.validators.operation.randomwalk_mode_to_sampler_config  # noqa: F401

        # Sample Store validators
        import orchestrator.core.legacy.validators.resource.entitysource_to_samplestore  # noqa: F401
        import orchestrator.core.legacy.validators.samplestore.entitysource_migrations  # noqa: F401
        import orchestrator.core.legacy.validators.samplestore.v1_to_v2_csv_migration  # noqa: F401
    except ImportError:
        pass  # Validators may not be available in all installations


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
        console.print(f"    Handles: {', '.join(validator.deprecated_fields)}")
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


# Made with Bob


def extract_deprecated_fields_from_validation_error(
    error: pydantic.ValidationError,
) -> tuple[set[str], dict[str, list[str]], set[str]]:
    """Extract field names and error details from pydantic validation errors

    Args:
        error: The pydantic validation error

    Returns:
        Tuple of (full field paths, field error details mapping, leaf field names)
        - full field paths: Set of full dotted paths like 'config.specification.module.moduleType'
        - field error details: Maps full field path to list of error messages
        - leaf field names: Set of just the final field names for validator matching
    """
    full_field_paths: set[str] = set()
    field_errors: dict[str, list[str]] = {}
    leaf_field_names: set[str] = set()

    for err in error.errors():
        if err.get("loc"):
            # Build the full dotted path from the location tuple
            full_path = ".".join(str(loc) for loc in err["loc"])
            full_field_paths.add(full_path)

            # Get the leaf field name (last element) for validator matching
            leaf_field = str(err["loc"][-1])
            leaf_field_names.add(leaf_field)

            # Store the error message for this field path
            if full_path not in field_errors:
                field_errors[full_path] = []

            # Build a descriptive error message
            msg = err.get("msg", "")
            if err.get("input"):
                msg = f"{msg} (got: {err['input']})"

            field_errors[full_path].append(msg)

    return full_field_paths, field_errors, leaf_field_names


def extract_deprecated_fields_from_value_error(
    error: ValueError,
) -> tuple[set[str], dict[str, list[str]], set[str]]:
    """Extract field names from ValueError containing pydantic validation errors

    This function attempts to extract the underlying pydantic ValidationError
    from a ValueError and extract field names from it. If that fails, it falls
    back to regex pattern matching on the error message.

    Args:
        error: The ValueError that may contain a pydantic ValidationError

    Returns:
        Tuple of (full field paths, field error details mapping, leaf field names)
    """
    # Try to extract pydantic ValidationError from the ValueError
    if hasattr(error, "__cause__") and isinstance(
        error.__cause__, pydantic.ValidationError
    ):
        return extract_deprecated_fields_from_validation_error(error.__cause__)

    # Fallback to regex pattern matching on error message
    import re

    error_msg = str(error)
    full_field_paths: set[str] = set()
    field_errors: dict[str, list[str]] = {}
    leaf_field_names: set[str] = set()

    # Pattern: field_name followed by validation error
    field_patterns = [
        r"kind\s*\n\s*Input should be",  # kind field
        r"moduleType\s*\n\s*Input should be",  # moduleType field
        r"moduleClass\s*\n\s*",  # moduleClass field
        r"moduleName\s*\n\s*",  # moduleName field
        r"constitutivePropertyColumns",  # constitutivePropertyColumns field
        r"propertyMap",  # propertyMap field
        r"entitySourceIdentifier",  # entitySourceIdentifier field
        r"properties\s*\n",  # properties field
        r"actuators\s*\n",  # actuators field
        r"mode\s*\n",  # mode field (for randomwalk)
    ]

    for pattern in field_patterns:
        if re.search(pattern, error_msg, re.IGNORECASE):
            # Extract the field name from the pattern
            field_name = pattern.split(r"\s")[0].split(r"\\")[0]
            full_field_paths.add(field_name)
            leaf_field_names.add(field_name)
            # For regex fallback, we don't have detailed error messages
            field_errors[field_name] = [
                "Field validation failed (details in error message)"
            ]

    return full_field_paths, field_errors, leaf_field_names
