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
    deprecated_field_paths=[
        "config.specification.module.moduleType",
        "config.copyFrom.0.module.moduleType",
    ],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Converts moduleType value from 'entity_source' to 'sample_store'",
)
def migrate_module_type(data: dict) -> dict:
    """Convert moduleType from entity_source to sample_store

    This validator checks for moduleType field within config.specification.module
    and config.copyFrom[].module, converting them from 'entity_source' to 'sample_store'.

    Old format:
        config:
            specification:
                module:
                    moduleType: "entity_source"
            copyFrom:
                - module:
                    moduleType: "entity_source"

    New format:
        config:
            specification:
                module:
                    moduleType: "sample_store"
            copyFrom:
                - module:
                    moduleType: "sample_store"

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Update config.specification.module.moduleType
    if has_nested_field(data, "config.specification.module.moduleType"):
        module_type = get_nested_value(data, "config.specification.module.moduleType")
        if module_type == "entity_source":
            set_nested_value(
                data, "config.specification.module.moduleType", "sample_store"
            )

    # Update config.copyFrom[].module.moduleType
    if has_nested_field(data, "config.copyFrom"):
        copy_from = get_nested_value(data, "config.copyFrom")
        if isinstance(copy_from, list):
            for item in copy_from:
                if isinstance(item, dict) and has_nested_field(
                    item, "module.moduleType"
                ):
                    module_type = get_nested_value(item, "module.moduleType")
                    if module_type == "entity_source":
                        set_nested_value(item, "module.moduleType", "sample_store")

    return data


@legacy_validator(
    identifier="samplestore_module_class_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_field_paths=[
        "config.specification.module.moduleClass",
        "config.copyFrom.0.module.moduleClass",
    ],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Converts moduleClass values from EntitySource to SampleStore naming (CSVEntitySource -> CSVSampleStore, SQLEntitySource -> SQLSampleStore)",
)
def migrate_module_class(data: dict) -> dict:
    """Convert moduleClass from EntitySource to SampleStore naming

    This validator checks for moduleClass field within config.specification.module
    and config.copyFrom[].module, converting them from EntitySource to SampleStore naming.

    Old format:
        config:
            specification:
                module:
                    moduleClass: "CSVEntitySource" or "SQLEntitySource"
            copyFrom:
                - module:
                    moduleClass: "CSVEntitySource"

    New format:
        config:
            specification:
                module:
                    moduleClass: "CSVSampleStore" or "SQLSampleStore"
            copyFrom:
                - module:
                    moduleClass: "CSVSampleStore"

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

    # Update config.specification.module.moduleClass
    if has_nested_field(data, "config.specification.module.moduleClass"):
        module_class = get_nested_value(data, "config.specification.module.moduleClass")
        if isinstance(module_class, str) and module_class in value_mappings:
            set_nested_value(
                data,
                "config.specification.module.moduleClass",
                value_mappings[module_class],
            )

    # Update config.copyFrom[].module.moduleClass
    if has_nested_field(data, "config.copyFrom"):
        copy_from = get_nested_value(data, "config.copyFrom")
        if isinstance(copy_from, list):
            for item in copy_from:
                if isinstance(item, dict) and has_nested_field(
                    item, "module.moduleClass"
                ):
                    module_class = get_nested_value(item, "module.moduleClass")
                    if isinstance(module_class, str) and module_class in value_mappings:
                        set_nested_value(
                            item, "module.moduleClass", value_mappings[module_class]
                        )

    return data


@legacy_validator(
    identifier="samplestore_module_name_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_field_paths=[
        "config.specification.module.moduleName",
        "config.copyFrom.0.module.moduleName",
    ],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Updates module paths from entitysource to samplestore (orchestrator.core.entitysource -> orchestrator.core.samplestore)",
)
def migrate_module_name(data: dict) -> dict:
    """Convert moduleName paths from entitysource to samplestore

    This validator checks for moduleName field within config.specification.module
    and config.copyFrom[].module, converting paths from entitysource to samplestore
    using substring replacement.

    Migrates any path containing the old module names:
        config:
            specification:
                module:
                    moduleName: "orchestrator.core.entitysource"
                    -> "orchestrator.core.samplestore"

                    moduleName: "orchestrator.plugins.entitysources"
                    -> "orchestrator.plugins.samplestores"

                    moduleName: "orchestrator.core.entitysource.csv"
                    -> "orchestrator.core.samplestore.csv"
            copyFrom:
                - module:
                    moduleName: "orchestrator.core.entitysource.sql"
                    -> "orchestrator.core.samplestore.sql"

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

    # Update config.specification.module.moduleName
    if has_nested_field(data, "config.specification.module.moduleName"):
        module_name = get_nested_value(data, "config.specification.module.moduleName")
        if isinstance(module_name, str):
            for old_path, new_path in path_mappings.items():
                if old_path in module_name:
                    set_nested_value(
                        data,
                        "config.specification.module.moduleName",
                        module_name.replace(old_path, new_path),
                    )
                    break

    # Update config.copyFrom[].module.moduleName
    if has_nested_field(data, "config.copyFrom"):
        copy_from = get_nested_value(data, "config.copyFrom")
        if isinstance(copy_from, list):
            for item in copy_from:
                if isinstance(item, dict) and has_nested_field(
                    item, "module.moduleName"
                ):
                    module_name = get_nested_value(item, "module.moduleName")
                    if isinstance(module_name, str):
                        for old_path, new_path in path_mappings.items():
                            if old_path in module_name:
                                set_nested_value(
                                    item,
                                    "module.moduleName",
                                    module_name.replace(old_path, new_path),
                                )
                                break

    return data


# Made with Bob
