# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for migrating CSV sample stores from v1 to v2 format"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.legacy.utils import get_nested_value, has_nested_field
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="csv_constitutive_columns_migration",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_field_paths=[
        "config.constitutivePropertyColumns",
        "config.experiments",
    ],
    deprecated_from_version="1.3.5",
    removed_from_version="1.6.0",
    description="Migrates CSV sample stores from v1 format (constitutivePropertyColumns in config) to v2 format (per-experiment constitutivePropertyMap)",
    dependencies=["samplestore_kind_entitysource_to_samplestore"],
)
def migrate_csv_v1_to_v2(data: dict) -> dict:
    """Migrate old CSVSampleStoreDescription format to new format

    This validator operates on the config section of the CSV sample store,
    migrating from v1 to v2 format. It matches the original pydantic
    validator behavior.

    Old format (in config):
        constitutivePropertyColumns: [...]
        experiments:
          - propertyMap: {...}

    New format (in config):
        # No constitutivePropertyColumns
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

    # Check if this is old format (has constitutivePropertyColumns in config)
    if not has_nested_field(data, "config.constitutivePropertyColumns"):
        return data

    # Get config value
    config = get_nested_value(data, "config")
    if config is None or not isinstance(config, dict):
        return data

    constitutive_columns = config.pop("constitutivePropertyColumns")
    # Migrate experiments if present in config
    experiments = config.get("experiments")
    if isinstance(experiments, list):
        for exp in experiments:
            if isinstance(exp, dict):
                # Rename propertyMap to observedPropertyMap
                if "propertyMap" in exp:
                    exp["observedPropertyMap"] = exp.pop("propertyMap")
                # Add constitutivePropertyMap from config-level constitutivePropertyColumns
                exp["constitutivePropertyMap"] = constitutive_columns

    return data


# Made with Bob
