# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for renaming entitySourceIdentifier to sampleStoreIdentifier"""

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.legacy.utils import (
    get_nested_value,
    remove_nested_field,
    set_nested_value,
)
from orchestrator.core.resources import CoreResourceKinds


@legacy_migrator(
    identifier="discoveryspace_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.DISCOVERYSPACE,
    deprecated_field_paths=["config.entitySourceIdentifier"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Renames 'entitySourceIdentifier' to 'sampleStoreIdentifier' in discovery space configurations",
)
def rename_entitysource_identifier(data: dict) -> dict:
    """Rename entitySourceIdentifier to sampleStoreIdentifier

    The 'entitySourceIdentifier' field was renamed to 'sampleStoreIdentifier' in config.
    This validator operates only on the config level, matching the original
    pydantic validator behavior.

    Old format:
        config:
            entitySourceIdentifier: "store-id"

    New format:
        config:
            sampleStoreIdentifier: "store-id"

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    old_path = "config.entitySourceIdentifier"
    new_path = "config.sampleStoreIdentifier"

    # Get the old value if it exists
    old_value = get_nested_value(data, old_path)
    if old_value is not None:
        set_nested_value(data, new_path, old_value)
        remove_nested_field(data, old_path)

    return data


# Made with Bob
