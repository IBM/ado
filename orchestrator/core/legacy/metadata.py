# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Metadata models for legacy validators"""

from collections.abc import Callable
from typing import Annotated

import pydantic

from orchestrator.core.resources import CoreResourceKinds


class LegacyValidatorMetadata(pydantic.BaseModel):
    """Metadata for a legacy validator function"""

    identifier: Annotated[
        str,
        pydantic.Field(
            description="Unique identifier for this validator (e.g., 'csv_constitutive_columns_migration')"
        ),
    ]

    resource_type: Annotated[
        CoreResourceKinds,
        pydantic.Field(description="Resource type this validator applies to"),
    ]

    deprecated_from_version: Annotated[
        str,
        pydantic.Field(description="ADO version when these fields were deprecated"),
    ]

    removed_from_version: Annotated[
        str,
        pydantic.Field(description="ADO version when automatic upgrade was removed"),
    ]

    description: Annotated[
        str,
        pydantic.Field(
            description="Human-readable description of what this validator does"
        ),
    ]

    validator_function: Annotated[
        Callable[[dict], dict],
        pydantic.Field(
            description="The actual migration function",
            exclude=True,  # Don't serialize the function
        ),
    ]

    deprecated_field_paths: Annotated[
        list[str],
        pydantic.Field(
            description="Explicit paths to fields (e.g., 'config.properties', 'config.specification.moduleType')"
        ),
    ]

    dependencies: Annotated[
        list[str],
        pydantic.Field(
            default_factory=list,
            description="List of validator identifiers that must run before this validator",
        ),
    ]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def format_info(
        self,
        index: int | None = None,
        show_dependencies: bool = True,
        show_version_info: bool = False,
    ) -> str:
        """Format validator information as a string

        Args:
            index: Optional index number to display (e.g., "1." for execution order)
            show_dependencies: Whether to show dependency information
            show_version_info: Whether to show version information

        Returns:
            Formatted string with validator information
        """
        from orchestrator.core.legacy.registry import LegacyValidatorRegistry

        lines = []

        # Validator identifier with optional index
        if index is not None:
            lines.append(f"  {index}. [green]{self.identifier}[/green]")
        else:
            lines.append(f"[green]{self.identifier}[/green]")

        # Description
        lines.append(f"     {self.description}")

        # Field paths
        lines.append(f"     Handles: {', '.join(self.deprecated_field_paths)}")

        # Dependencies
        if show_dependencies and self.dependencies:
            dep_names = []
            for dep_id in self.dependencies:
                dep_validator = LegacyValidatorRegistry.get_validator(dep_id)
                if dep_validator:
                    dep_names.append(dep_validator.identifier)
                else:
                    dep_names.append(f"{dep_id} [red](missing)[/red]")
            lines.append(f"     Depends on: {', '.join(dep_names)}")

        # Version information
        if show_version_info:
            lines.append(
                f"     Deprecated from: [cyan]{self.deprecated_from_version}[/cyan]"
            )
            lines.append(f"     Removed from: [cyan]{self.removed_from_version}[/cyan]")

        return "\n".join(lines)


# Made with Bob
