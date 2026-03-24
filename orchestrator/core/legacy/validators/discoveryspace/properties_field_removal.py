# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for removing deprecated properties field from discovery spaces"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="discoveryspace_properties_field_removal",
    resource_type=CoreResourceKinds.DISCOVERYSPACE,
    deprecated_fields=["properties"],
    deprecated_from_version="0.10.1",
    removed_from_version="1.0.0",
    description="Removes the deprecated 'properties' field from discovery space configurations",
)
def remove_properties_field(data: dict) -> dict:
    """Remove deprecated properties field from discovery space configuration

    The 'properties' field was deprecated in config and should be removed.
    This validator operates only on the config level, matching the original
    pydantic validator behavior.

    Old format:
        config:
            properties: [...]

    New format:
        config:
            # No properties field

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Only check in config (where the field actually exists)
    if "config" in data and isinstance(data["config"], dict):
        data["config"].pop("properties", None)

    return data


# Made with Bob
