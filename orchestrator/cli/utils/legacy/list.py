# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utilities for listing legacy migrators"""

from orchestrator.cli.utils.output.prints import console_print
from orchestrator.core.legacy.registry import LegacyMigratorRegistry
from orchestrator.core.resources import CoreResourceKinds


def list_legacy_migrators(resource_type: CoreResourceKinds) -> None:
    """List all available legacy migrators for a specific resource type

    Args:
        resource_type: The resource type to list migrators for
    """
    # Import migrators package to trigger registration via __init__.py
    import orchestrator.core.legacy.migrators  # noqa: F401

    # Get migrators for this resource type
    migrators = LegacyMigratorRegistry.get_migrators_for_resource(resource_type)

    if not migrators:
        console_print(
            f"\n[yellow]No legacy migrators available for {resource_type.value}[/yellow]\n"
        )
        return

    # Resources can be referenced by their CoreResourceKinds value or by shorthands
    # from cli_shorthands_to_cli_names in orchestrator/cli/utils/resources/mappings.py
    resource_cli_name = resource_type.value

    console_print(f"Available legacy migrators for {resource_cli_name}s:\n")

    for i, migrator in enumerate(migrators, 1):
        # Format and print migrator info with version information using the method
        console_print(
            migrator.format_info(
                index=i, show_dependencies=True, show_version_info=False
            )
        )
        console_print()  # Add spacing between migrators


# Made with Bob
