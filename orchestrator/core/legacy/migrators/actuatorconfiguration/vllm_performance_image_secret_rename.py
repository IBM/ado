# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for vllm_performance actuator image_secret field rename"""

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.legacy.utils import (
    get_nested_value,
    has_nested_field,
    remove_nested_field,
    set_nested_value,
)
from orchestrator.core.resources import CoreResourceKinds


@legacy_migrator(
    identifier="vllm_performance_image_secret_rename",
    resource_type=CoreResourceKinds.ACTUATORCONFIGURATION,
    deprecated_field_paths=["parameters.image_secret"],
    deprecated_from_version="1.4.1",
    removed_from_version="1.7.0",
    description="Renames 'image_secret' to 'image_pull_secret_name' in vllm_performance actuator parameters",
)
def rename_image_secret_field(data: dict) -> dict:
    """Rename image_secret to image_pull_secret_name in vllm_performance actuator

    This migrator handles the rename of the deprecated 'image_secret' field to
    'image_pull_secret_name' in vllm_performance actuator configurations.

    Old format:
        actuatorIdentifier: vllm_performance
        parameters:
            image_secret: "my-secret"

    New format:
        actuatorIdentifier: vllm_performance
        parameters:
            image_pull_secret_name: "my-secret"

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """
    if not isinstance(data, dict):
        return data

    # Only apply to vllm_performance actuator
    actuator_id = data.get("actuatorIdentifier")
    if actuator_id != "vllm_performance":
        return data

    old_path = "parameters.image_secret"
    new_path = "parameters.image_pull_secret_name"

    # Check if old field exists
    if not has_nested_field(data, old_path):
        return data

    # Get the old value
    old_value = get_nested_value(data, old_path)

    # If new field already exists, remove old field (new takes precedence)
    if has_nested_field(data, new_path):
        remove_nested_field(data, old_path)
    else:
        # Set new field with old value and remove old field
        set_nested_value(data, new_path, old_value)
        remove_nested_field(data, old_path)

    return data


# Made with Bob
