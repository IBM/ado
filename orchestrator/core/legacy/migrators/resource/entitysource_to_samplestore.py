# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for migrating entitysource kind to samplestore kind"""

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.utilities.dictionaries import has_nested_field, set_nested_value


@legacy_migrator(
    identifier="samplestore_kind_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_field_paths=["kind"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Converts resource kind from 'entitysource' to 'samplestore'",
    dependencies=[
        "samplestore_module_type_entitysource_to_samplestore",
        "samplestore_module_class_entitysource_to_samplestore",
        "samplestore_module_name_entitysource_to_samplestore",
    ],
)
def migrate_entitysource_kind_to_samplestore(data: dict) -> dict:
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
    if has_nested_field(data, "kind") and data.get("kind") == "entitysource":
        set_nested_value(data, "kind", "samplestore")

    return data


# Made with Bob
