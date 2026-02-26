# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for migrating entitysource kind to samplestore kind"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_fields=["kind"],
    deprecated_from_version="1.2.0",
    removed_from_version="1.5.0",
    description="Migrates resources with kind='entitysource' to kind='samplestore'",
)
def migrate_entitysource_to_samplestore(data: dict) -> dict:
    """Migrate old entitysource kind to samplestore

    Old format:
        kind: "entitysource"

    New format:
        kind: "samplestore"

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Check if this is an entitysource that needs migration
    if data.get("kind") == "entitysource":
        data["kind"] = "samplestore"

    return data


# Made with Bob
