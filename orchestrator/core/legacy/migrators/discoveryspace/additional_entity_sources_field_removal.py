# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for removing deprecated additionalEntitySources field from discovery spaces"""

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.utilities.dictionaries import remove_nested_field


@legacy_migrator(
    identifier="discoveryspace_additional_entity_sources_field_removal",
    resource_type=CoreResourceKinds.DISCOVERYSPACE,
    deprecated_field_paths=["config.additionalEntitySources"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Removes the deprecated 'additionalEntitySources' field from discovery space configurations.",
)
def remove_additional_entity_sources_field(data: dict) -> dict:
    """Remove deprecated config.additionalEntitySources from discovery space configuration.

    Args:
        data: The resource data dictionary.

    Returns:
        The migrated resource data dictionary.
    """
    if not isinstance(data, dict):
        return data

    remove_nested_field(data, "config.additionalEntitySources")
    return data
