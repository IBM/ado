# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for removing deprecated spaceIdentifier field from operations"""

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.utilities.dictionaries import remove_nested_field


@legacy_migrator(
    identifier="operation_space_identifier_field_removal",
    resource_type=CoreResourceKinds.OPERATION,
    deprecated_field_paths=["config.spaceIdentifier"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Removes the deprecated 'spaceIdentifier' field from operation configurations when 'spaces' is present.",
)
def remove_space_identifier_field(data: dict) -> dict:
    """Remove deprecated config.spaceIdentifier from operation configuration.

    Args:
        data: The resource data dictionary.

    Returns:
        The migrated resource data dictionary.
    """
    if not isinstance(data, dict):
        return data

    remove_nested_field(data, "config.spaceIdentifier")
    return data
