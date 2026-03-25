# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utilities for listing legacy validators"""

from orchestrator.cli.utils.output.prints import console_print
from orchestrator.core.legacy.registry import LegacyValidatorRegistry
from orchestrator.core.resources import CoreResourceKinds


def list_legacy_validators(resource_type: CoreResourceKinds) -> None:
    """List all available legacy validators for a specific resource type

    Args:
        resource_type: The resource type to list validators for
    """
    # Import validators package to trigger registration via __init__.py
    import orchestrator.core.legacy.validators  # noqa: F401

    # Get validators for this resource type
    validators = LegacyValidatorRegistry.get_validators_for_resource(resource_type)

    if not validators:
        console_print(
            f"\n[yellow]No legacy validators available for {resource_type.value}[/yellow]\n"
        )
        return

    # Resources can be referenced by their CoreResourceKinds value or by shorthands
    # from cli_shorthands_to_cli_names in orchestrator/cli/utils/resources/mappings.py
    resource_cli_name = resource_type.value

    console_print(f"Available legacy validators for {resource_cli_name}s:\n")

    for i, validator in enumerate(validators, 1):
        # Format and print validator info with version information using the method
        console_print(
            validator.format_info(
                index=i, show_dependencies=True, show_version_info=False
            )
        )
        console_print()  # Add spacing between validators


# Made with Bob
