# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for renaming entitySourceIdentifier to sampleStoreIdentifier"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="discoveryspace_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.DISCOVERYSPACE,
    deprecated_fields=["entitySourceIdentifier"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Renames 'entitySourceIdentifier' to 'sampleStoreIdentifier' in discovery space configurations",
)
def rename_entitysource_identifier(data: dict) -> dict:
    """Rename entitySourceIdentifier to sampleStoreIdentifier

    Old format:
        - Used 'entitySourceIdentifier' field

    New format:
        - Uses 'sampleStoreIdentifier' field

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    old_key = "entitySourceIdentifier"
    new_key = "sampleStoreIdentifier"

    # Check at top level
    if old_key in data:
        data[new_key] = data.pop(old_key)

    # Also check in config if present
    if (
        "config" in data
        and isinstance(data["config"], dict)
        and old_key in data["config"]
    ):
        data["config"][new_key] = data["config"].pop(old_key)

    return data


# Made with Bob
