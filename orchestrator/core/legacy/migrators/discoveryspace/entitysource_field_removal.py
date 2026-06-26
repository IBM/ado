# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for removing deprecated entitySource field from discovery spaces"""

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.utilities.dictionaries import (
    get_nested_value,
    remove_nested_field,
    set_nested_value,
)


@legacy_migrator(
    identifier="discoveryspace_entitysource_field_removal",
    resource_type=CoreResourceKinds.DISCOVERYSPACE,
    deprecated_field_paths=["config.entitySource"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Removes deprecated 'entitySource' field from discovery space configurations and promotes its identifier to 'sampleStoreIdentifier'.",
)
def remove_entitysource_field(data: dict) -> dict:
    """Remove config.entitySource and preserve its referenced sample store identifier.

    Args:
        data: The resource data dictionary.

    Returns:
        The migrated resource data dictionary.
    """
    if not isinstance(data, dict):
        return data

    sample_store_identifier = get_nested_value(
        data, "config.entitySource.parameters.identifier"
    )
    if (
        sample_store_identifier is not None
        and get_nested_value(data, "config.sampleStoreIdentifier") is None
    ):
        set_nested_value(data, "config.sampleStoreIdentifier", sample_store_identifier)

    remove_nested_field(data, "config.entitySource")
    return data
