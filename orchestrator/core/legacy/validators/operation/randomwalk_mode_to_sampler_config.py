# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for migrating random_walk parameters to samplerConfig"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.legacy.utils import (
    get_nested_value,
    has_nested_field,
    remove_nested_field,
    set_nested_value,
)
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="randomwalk_mode_to_sampler_config",
    resource_type=CoreResourceKinds.OPERATION,
    deprecated_fields=["mode", "grouping", "samplerType"],
    deprecated_from_version="1.0.1",
    removed_from_version="1.2",
    description="Migrates random_walk parameters from flat structure to nested 'samplerConfig'. See https://ibm.github.io/ado/operators/random-walk/#configuring-a-randomwalk",
    field_paths=[
        "config.parameters.mode",
        "config.parameters.grouping",
        "config.parameters.samplerType",
    ],
)
def migrate_randomwalk_to_sampler_config(data: dict) -> dict:
    """Migrate random_walk parameters to samplerConfig structure

    Old format:
        - mode, grouping, samplerType at top level of parameters

    New format:
        - These fields nested under samplerConfig

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Check if mode field exists (indicator of old format)
    if not has_nested_field(data, "config.parameters.mode"):
        return data

    # Extract the old fields - has_nested_field already confirmed they exist
    mode = None
    grouping = None
    sampler_type = None

    if has_nested_field(data, "config.parameters.mode"):
        parent, field = get_nested_value(data, "config.parameters.mode")
        if parent is not None:
            mode = parent[field]
            remove_nested_field(data, "config.parameters.mode")

    if has_nested_field(data, "config.parameters.grouping"):
        parent, field = get_nested_value(data, "config.parameters.grouping")
        if parent is not None:
            grouping = parent[field]
            remove_nested_field(data, "config.parameters.grouping")

    if has_nested_field(data, "config.parameters.samplerType"):
        parent, field = get_nested_value(data, "config.parameters.samplerType")
        if parent is not None:
            sampler_type = parent[field]
            remove_nested_field(data, "config.parameters.samplerType")

    # Create samplerConfig if any of the fields were present
    if mode is not None or grouping is not None or sampler_type is not None:
        sampler_config = {}
        if mode is not None:
            sampler_config["mode"] = mode
        if grouping is not None:
            sampler_config["grouping"] = grouping
        if sampler_type is not None:
            sampler_config["samplerType"] = sampler_type

        set_nested_value(data, "config.parameters.samplerConfig", sampler_config)

    return data


# Made with Bob
