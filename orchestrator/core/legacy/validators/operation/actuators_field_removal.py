# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for removing deprecated actuators field from operations"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="operation_actuators_field_removal",
    resource_type=CoreResourceKinds.OPERATION,
    deprecated_fields=["actuators"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Removes the deprecated 'actuators' field from operation configurations. See https://ibm.github.io/ado/resources/operation/#the-operation-configuration-yaml",
)
def remove_actuators_field(data: dict) -> dict:
    """Remove deprecated actuators field from operation configuration

    Old format:
        - Had 'actuators' field in config

    New format:
        - No 'actuators' field (use actuator configurations instead)

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Remove actuators field if present at top level
    if "actuators" in data:
        data.pop("actuators", None)

    # Also check in config if present
    if (
        "config" in data
        and isinstance(data["config"], dict)
        and "actuators" in data["config"]
    ):
        data["config"].pop("actuators", None)

    return data


# Made with Bob
