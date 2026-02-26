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

    Old format:
        - constitutivePropertyColumns at top level (list)
        - experiments list with propertyMap (not observedPropertyMap)
        - No constitutivePropertyMap in experiment descriptions

    New format:
        - No constitutivePropertyColumns at top level
        - experiments with observedPropertyMap and constitutivePropertyMap

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Check if this is old format (has constitutivePropertyColumns at top level)
    if "constitutivePropertyColumns" not in data:
        return data

    # Extract and remove the top-level constitutivePropertyColumns
    constitutive_columns = data.pop("constitutivePropertyColumns")

    # Migrate experiments if present
    if "experiments" in data and isinstance(data["experiments"], list):
        for exp in data["experiments"]:
            if isinstance(exp, dict):
                # Rename propertyMap to observedPropertyMap
                if "propertyMap" in exp:
                    exp["observedPropertyMap"] = exp.pop("propertyMap")
                # Add constitutivePropertyMap from top-level constitutivePropertyColumns
                exp["constitutivePropertyMap"] = constitutive_columns

    return data


# Made with Bob
