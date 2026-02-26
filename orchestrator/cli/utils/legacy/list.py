# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utilities for listing legacy validators"""

from rich.console import Console
from rich.panel import Panel

from orchestrator.core.legacy.registry import LegacyValidatorRegistry
from orchestrator.core.resources import CoreResourceKinds


def _import_legacy_validators() -> None:
    """Import all legacy validator modules to ensure they're registered"""
    # Import validator modules to trigger decorator registration
    try:
        import orchestrator.core.legacy.validators.resource.entitysource_to_samplestore  # noqa: F401
        import orchestrator.core.legacy.validators.samplestore.v1_to_v2_csv_migration  # noqa: F401
    except ImportError:
        pass  # Validators may not be available in all installations


def list_legacy_validators(resource_type: CoreResourceKinds) -> None:
    """List all available legacy validators for a specific resource type

    Args:
        resource_type: The resource type to list validators for
    """
    console = Console()

    # Import all validator modules to ensure they're registered
    _import_legacy_validators()

    # Get validators for this resource type
    validators = LegacyValidatorRegistry.get_validators_for_resource(resource_type)

    if not validators:
        console.print(
            f"\n[yellow]No legacy validators available for {resource_type.value}[/yellow]\n"
        )
        return

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

    console.print(
        f"\n[bold cyan]Available legacy validators for {resource_cli_name}:[/bold cyan]\n"
    )

    for validator in validators:
        # Create a panel for each validator
        content_lines = []

        # Description
        content_lines.append("[bold]Description:[/bold]")
        content_lines.append(f"  {validator.description}")
        content_lines.append("")

        # Deprecated fields
        content_lines.append("[bold]Handles deprecated fields:[/bold]")
        content_lines.extend(f"  • {field}" for field in validator.deprecated_fields)
        content_lines.append("")

        # Version info
        content_lines.append("[bold]Version info:[/bold]")
        content_lines.append(
            f"  Deprecated from: [cyan]{validator.deprecated_from_version}[/cyan]"
        )
        content_lines.append(
            f"  Removed from: [cyan]{validator.removed_from_version}[/cyan]"
        )
        content_lines.append("")

        # Usage
        content_lines.append("[bold]Usage:[/bold]")
        content_lines.append(
            f"  [green]ado upgrade {resource_cli_name} --apply-legacy-validator {validator.identifier}[/green]"
        )

        panel = Panel(
            "\n".join(content_lines),
            title=f"[bold magenta]{validator.identifier}[/bold magenta]",
            border_style="cyan",
            expand=False,
        )
        console.print(panel)
        console.print()  # Add spacing between panels

    console.print(
        f"[bold]Found {len(validators)} legacy validator(s) for {resource_cli_name}[/bold]\n"
    )


# Made with Bob
