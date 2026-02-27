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

    Old format:
        - Had 'properties' field at top level

    New format:
        - No 'properties' field

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Remove properties field if present
    if "properties" in data:
        data.pop("properties", None)

    # Also check in config if present
    if (
        "config" in data
        and isinstance(data["config"], dict)
        and "properties" in data["config"]
    ):
        data["config"].pop("properties", None)

    return data


# Made with Bob
