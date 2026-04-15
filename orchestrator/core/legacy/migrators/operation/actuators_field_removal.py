# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for removing deprecated actuators field from operations"""

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.utilities.dictionaries import remove_nested_field


@legacy_migrator(
    identifier="operation_actuators_field_removal",
    resource_type=CoreResourceKinds.OPERATION,
    deprecated_field_paths=["config.actuators"],
    deprecated_from_version="0.9.6",
    removed_from_version="1.0.0",
    description="Removes the deprecated 'actuators' field from operation configurations. See https://ibm.github.io/ado/resources/operation/#the-operation-configuration-yaml",
)
def remove_actuators_field(data: dict) -> dict:
    """Remove deprecated actuators field from operation configuration

    The 'actuators' field was deprecated in config and should be removed.
    This validator operates only on the config level, matching the original
    pydantic validator behavior.

    Old format:
        config:
            actuators: [...]

    New format:
        config:
            # No actuators field (use actuator configurations instead)

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """
    if isinstance(data, dict):
        remove_nested_field(data, "config.actuators")

    return data


# Made with Bob
