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
    deprecated_field_paths=[
        "config.parameters.mode",
        "config.parameters.grouping",
        "config.parameters.samplerType",
    ],
    deprecated_from_version="1.0.1",
    removed_from_version="1.2",
    description="Migrates random_walk parameters from flat structure to nested 'samplerConfig'. See https://ibm.github.io/ado/operators/random-walk/#configuring-a-randomwalk",
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

    # Fields to migrate from top-level parameters to samplerConfig
    fields_to_migrate = ["mode", "grouping", "samplerType"]

    sampler_config = {}

    # Extract and migrate each field
    for field_name in fields_to_migrate:
        field_path = f"config.parameters.{field_name}"
        if has_nested_field(data, field_path):
            field_value = get_nested_value(data, field_path)
            if field_value is not None:
                sampler_config[field_name] = field_value
                remove_nested_field(data, field_path)

    # Only set samplerConfig if we found any fields to migrate
    if sampler_config:
        set_nested_value(data, "config.parameters.samplerConfig", sampler_config)

    return data


# Made with Bob
