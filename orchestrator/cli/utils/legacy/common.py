# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Common utilities for legacy migrator handling"""

from typing import TYPE_CHECKING

import pydantic

from orchestrator.cli.utils.output.prints import (
    ERROR,
    HINT,
    INFO,
    WARN,
    console_print,
    cyan,
)

if TYPE_CHECKING:
    from orchestrator.core.legacy.metadata import LegacyMigratorMetadata
    from orchestrator.core.resources import CoreResourceKinds


def extract_deprecated_field_paths(
    error: pydantic.ValidationError | ValueError,
    resource_type: "CoreResourceKinds | None" = None,
) -> tuple[set[str], dict[str, list[str]]]:
    """Extract field paths and error details from validation errors

    This function handles both pydantic ValidationError and ValueError types.
    For ValueError, it attempts to extract an underlying pydantic ValidationError
    from the error's __cause__. If that fails, it falls back to simple string
    matching on the error message using known field paths from the legacy
    migrator registry (requires resource_type parameter).

    Args:
        error: The validation error (pydantic.ValidationError or ValueError)
        resource_type: The resource type to get field paths for (required for
            ValueError fallback to string matching)

    Returns:
        Tuple of (full field paths, field error details mapping)
        - full field paths: Set of full dotted paths like 'config.specification.module.moduleType'
        - field error details: Maps full field path to list of error messages

    Raises:
        ValueError: Re-raises the original error if it is a ValueError without a
            ValidationError cause and resource_type is not provided
    """
    deprecated_field_paths: set[str] = set()
    field_errors: dict[str, list[str]] = {}

    # Handle pydantic ValidationError directly
    if isinstance(error, pydantic.ValidationError):
        for err in error.errors():
            if err.get("loc"):
                # Build the full dotted path from the location tuple
                full_path = ".".join(str(loc) for loc in err["loc"])
                deprecated_field_paths.add(full_path)

                # Store the error message for this field path
                if full_path not in field_errors:
                    field_errors[full_path] = []

                # Build a descriptive error message
                msg = err.get("msg", "")
                if err.get("input"):
                    msg = f"{msg} (got: {err['input']})"

                field_errors[full_path].append(msg)

        return deprecated_field_paths, field_errors

    # Handle ValueError - try to extract pydantic ValidationError from __cause__
    if isinstance(error, ValueError):
        if hasattr(error, "__cause__") and isinstance(
            error.__cause__, pydantic.ValidationError
        ):
            # Recursively handle the underlying ValidationError
            return extract_deprecated_field_paths(error.__cause__, resource_type)

        # Fallback to simple string matching on error message
        if resource_type is None:
            raise error

        from orchestrator.core.legacy.registry import LegacyMigratorRegistry

        error_msg = str(error)

        # Get all field paths from registered migrators for this resource type
        migrators = LegacyMigratorRegistry.get_migrators_for_resource(resource_type)
        known_deprecated_field_paths = {
            path for migrator in migrators for path in migrator.deprecated_field_paths
        }

        for field_path in known_deprecated_field_paths:
            if field_path in error_msg:
                deprecated_field_paths.add(field_path)
                # For string matching fallback, we don't have detailed error messages
                field_errors[field_path] = [
                    "Field validation failed (details in error message)"
                ]

        return deprecated_field_paths, field_errors

    # Should not reach here due to type hints, but handle gracefully
    raise TypeError(f"Unsupported error type: {type(error)}")


def print_migrator_suggestions_with_dependencies(
    migrators: list["LegacyMigratorMetadata"], resource_type: "CoreResourceKinds"
) -> None:
    """Print legacy migrator suggestions with dependency information

    This enhanced version resolves dependencies and shows migrators in the
    correct execution order, along with dependency information.

    Args:
        migrators: List of applicable migrators
        resource_type: The resource type
    """
    from orchestrator.core.legacy.registry import LegacyMigratorRegistry

    # Get migrator identifiers
    migrator_ids = [v.identifier for v in migrators]
    missing_deps = []

    # Resolve dependencies to get correct order
    try:
        migrator_ids, missing_deps = LegacyMigratorRegistry.resolve_dependencies(
            migrator_ids
        )
    except ValueError as e:
        # Circular dependency detected
        console_print(f"{ERROR}:{e}", stderr=True)

    # Get ordered migrators (filter out None values)
    ordered_migrators: list[LegacyMigratorMetadata] = []
    for vid in migrator_ids:
        migrator = LegacyMigratorRegistry.get_migrator(vid)
        if migrator is not None:
            ordered_migrators.append(migrator)

    console_print(f"{INFO}The following migrator(s) are a match:\n", stderr=True)
    for i, migrator in enumerate(ordered_migrators, 1):
        # Format and print migrator info using the method
        console_print(
            migrator.format_info(
                index=i, show_dependencies=True, show_version_info=False
            )
        )
        console_print()

    # Warn about missing dependencies
    if missing_deps:
        console_print(
            f"{WARN}Some dependencies are missing: {', '.join(missing_deps)}\n"
        )

    # Build command with all migrators in correct order
    migrator_args = " ".join(
        f"--apply-legacy-migrator {v.identifier}" for v in ordered_migrators
    )
    console_print(
        f"{HINT}To attempt the upgrade using the suggested legacy migrator(s) run:\n"
        f"\t{cyan(f'ado upgrade {resource_type.value} {migrator_args}')}\n"
    )

    # Show note about automatic dependency resolution
    if len(ordered_migrators) > len(migrators):
        console_print(
            "[dim]Note: Additional migrators were included to satisfy dependencies[/dim]\n"
        )

    console_print(
        f"{HINT}To list all legacy migrators run:\n"
        f"\t{cyan(f'ado upgrade {resource_type.value} --list-legacy-migrators')}"
    )
