# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for migrating CSV sample stores from v1 to v2 format"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="csv_constitutive_columns_migration",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_fields=["constitutivePropertyColumns", "propertyMap"],
    deprecated_from_version="1.3.5",
    removed_from_version="1.6.0",
    description="Migrates CSV sample stores from v1 format (constitutivePropertyColumns at top level) to v2 format (per-experiment constitutivePropertyMap)",
)
def migrate_csv_v1_to_v2(data: dict) -> dict:
    """Migrate old CSVSampleStoreDescription format to new format

    This validator operates on the config level, migrating CSV sample store
    configurations from v1 to v2 format. It matches the original pydantic
    validator behavior.

    Old format:
        config:
            constitutivePropertyColumns: [...]  # at config level
            experiments:
              - propertyMap: {...}

    New format:
        config:
            # No constitutivePropertyColumns at config level
            experiments:
              - observedPropertyMap: {...}
                constitutivePropertyMap: [...]

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

    config = data["config"]

    # Check if this is old format (has constitutivePropertyColumns in config)
    if "constitutivePropertyColumns" not in config:
        return data

    # Extract and remove the constitutivePropertyColumns from config
    constitutive_columns = config.pop("constitutivePropertyColumns")

    # Migrate experiments if present in config
    if "experiments" in config and isinstance(config["experiments"], list):
        for exp in config["experiments"]:
            if isinstance(exp, dict):
                # Rename propertyMap to observedPropertyMap
                if "propertyMap" in exp:
                    exp["observedPropertyMap"] = exp.pop("propertyMap")
                # Add constitutivePropertyMap from config-level constitutivePropertyColumns
                exp["constitutivePropertyMap"] = constitutive_columns

    return data


# Made with Bob
