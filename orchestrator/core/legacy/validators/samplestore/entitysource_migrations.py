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
        parent, field = get_nested_value(data, "config.moduleType")
        if parent is not None and parent[field] == "entity_source":
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
        parent, field = get_nested_value(data, "config.moduleClass")
        if parent is not None and parent[field] in value_mappings:
            set_nested_value(data, "config.moduleClass", value_mappings[parent[field]])

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
    and converts paths from entitysource to samplestore.

    Old format:
        config:
            moduleName: "orchestrator.core.entitysource.*"
            moduleName: "orchestrator.plugins.entitysources.*"

    New format:
        config:
            moduleName: "orchestrator.core.samplestore.*"
            moduleName: "orchestrator.plugins.samplestores.*"

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
        parent, field = get_nested_value(data, "config.moduleName")
        if parent is not None and isinstance(parent[field], str):
            module_name = parent[field]
            for old_path, new_path in path_mappings.items():
                if old_path in module_name:
                    module_name = module_name.replace(old_path, new_path)
            set_nested_value(data, "config.moduleName", module_name)

    return data


# Made with Bob
