# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validators for migrating entitysource to samplestore naming"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="samplestore_module_type_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_fields=["moduleType"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Converts moduleType value from 'entity_source' to 'sample_store'",
)
def migrate_module_type(data: dict) -> dict:
    """Convert moduleType from entity_source to sample_store

    This validator recursively searches for moduleType fields within the config
    and converts them from 'entity_source' to 'sample_store'. It operates only
    within the config level, matching the original pydantic validator behavior.

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

    # Only operate within config
    if "config" not in data or not isinstance(data["config"], dict):
        return data

    def convert_module_type_in_dict(d: dict) -> None:
        """Recursively convert moduleType in nested structures"""
        if "moduleType" in d and d["moduleType"] == "entity_source":
            d["moduleType"] = "sample_store"

        # Check in nested structures
        for value in d.values():
            if isinstance(value, dict):
                convert_module_type_in_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        convert_module_type_in_dict(item)

    # Start recursion from config level
    convert_module_type_in_dict(data["config"])
    return data


@legacy_validator(
    identifier="samplestore_module_class_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_fields=["moduleClass"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Converts moduleClass values from EntitySource to SampleStore naming (CSVEntitySource -> CSVSampleStore, SQLEntitySource -> SQLSampleStore)",
)
def migrate_module_class(data: dict) -> dict:
    """Convert moduleClass from EntitySource to SampleStore naming

    This validator recursively searches for moduleClass fields within the config
    and converts them from EntitySource to SampleStore naming. It operates only
    within the config level, matching the original pydantic validator behavior.

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

    # Only operate within config
    if "config" not in data or not isinstance(data["config"], dict):
        return data

    value_mappings = {
        "CSVEntitySource": "CSVSampleStore",
        "SQLEntitySource": "SQLSampleStore",
    }

    def convert_module_class_in_dict(d: dict) -> None:
        """Recursively convert moduleClass in nested structures"""
        if "moduleClass" in d and d["moduleClass"] in value_mappings:
            d["moduleClass"] = value_mappings[d["moduleClass"]]

        # Check in nested structures
        for value in d.values():
            if isinstance(value, dict):
                convert_module_class_in_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        convert_module_class_in_dict(item)

    # Start recursion from config level
    convert_module_class_in_dict(data["config"])
    return data


@legacy_validator(
    identifier="samplestore_module_name_entitysource_to_samplestore",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_fields=["moduleName"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Updates module paths from entitysource to samplestore (orchestrator.core.entitysource -> orchestrator.core.samplestore)",
)
def migrate_module_name(data: dict) -> dict:
    """Convert moduleName paths from entitysource to samplestore

    This validator recursively searches for moduleName fields within the config
    and converts paths from entitysource to samplestore. It operates only
    within the config level, matching the original pydantic validator behavior.

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

    # Only operate within config
    if "config" not in data or not isinstance(data["config"], dict):
        return data

    path_mappings = {
        "orchestrator.core.entitysource": "orchestrator.core.samplestore",
        "orchestrator.plugins.entitysources": "orchestrator.plugins.samplestores",
    }

    def convert_module_name_in_dict(d: dict) -> None:
        """Recursively convert moduleName in nested structures"""
        if "moduleName" in d and isinstance(d["moduleName"], str):
            for old_path, new_path in path_mappings.items():
                if old_path in d["moduleName"]:
                    d["moduleName"] = d["moduleName"].replace(old_path, new_path)
                    break

        # Check in nested structures
        for value in d.values():
            if isinstance(value, dict):
                convert_module_name_in_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        convert_module_name_in_dict(item)

    # Start recursion from config level
    convert_module_name_in_dict(data["config"])
    return data


# Made with Bob
