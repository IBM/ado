# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validators for migrating entitysource to samplestore naming"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.legacy.utils import (
    get_nested_value,
    has_nested_field,
    set_nested_value,
)
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="samplestore_module_type_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_field_paths=["config.moduleType"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Converts moduleType value from 'entity_source' to 'sample_store'",
)
def migrate_module_type(data: dict) -> dict:
    """Convert moduleType from entity_source to sample_store

    This validator checks for moduleType field within the config
    and converts it from 'entity_source' to 'sample_store'.

    Old format:
        config:
            moduleType: "entity_source"

    New format:
        config:
            moduleType: "sample_store"

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Check and update config.moduleType
    if has_nested_field(data, "config.moduleType"):
        module_type = get_nested_value(data, "config.moduleType")
        if module_type == "entity_source":
            set_nested_value(data, "config.moduleType", "sample_store")

    return data


@legacy_validator(
    identifier="samplestore_module_class_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_field_paths=["config.moduleClass"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Converts moduleClass values from EntitySource to SampleStore naming (CSVEntitySource -> CSVSampleStore, SQLEntitySource -> SQLSampleStore)",
)
def migrate_module_class(data: dict) -> dict:
    """Convert moduleClass from EntitySource to SampleStore naming

    This validator checks for moduleClass field within the config
    and converts it from EntitySource to SampleStore naming.

    Old format:
        config:
            moduleClass: "CSVEntitySource" or "SQLEntitySource"

    New format:
        config:
            moduleClass: "CSVSampleStore" or "SQLSampleStore"

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    value_mappings = {
        "CSVEntitySource": "CSVSampleStore",
        "SQLEntitySource": "SQLSampleStore",
    }

    # Check and update config.moduleClass
    if has_nested_field(data, "config.moduleClass"):
        module_class = get_nested_value(data, "config.moduleClass")
        if isinstance(module_class, str) and module_class in value_mappings:
            set_nested_value(data, "config.moduleClass", value_mappings[module_class])

    return data


@legacy_validator(
    identifier="samplestore_module_name_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_field_paths=["config.moduleName"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Updates module paths from entitysource to samplestore (orchestrator.core.entitysource -> orchestrator.core.samplestore)",
)
def migrate_module_name(data: dict) -> dict:
    """Convert moduleName paths from entitysource to samplestore

    This validator checks for moduleName field within the config
    and converts paths from entitysource to samplestore using exact matching.

    Only exact matches are migrated:
        config:
            moduleName: "orchestrator.core.entitysource"
            -> "orchestrator.core.samplestore"

            moduleName: "orchestrator.plugins.entitysources"
            -> "orchestrator.plugins.samplestores"

    Submodules or partial matches are NOT migrated:
        config:
            moduleName: "orchestrator.core.entitysource.csv"
            -> unchanged (not an exact match)

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    path_mappings = {
        "orchestrator.core.entitysource": "orchestrator.core.samplestore",
        "orchestrator.plugins.entitysources": "orchestrator.plugins.samplestores",
    }

    # Check and update config.moduleName
    if has_nested_field(data, "config.moduleName"):
        module_name = get_nested_value(data, "config.moduleName")
        if isinstance(module_name, str) and module_name in path_mappings:
            set_nested_value(data, "config.moduleName", path_mappings[module_name])

    return data


# Made with Bob
