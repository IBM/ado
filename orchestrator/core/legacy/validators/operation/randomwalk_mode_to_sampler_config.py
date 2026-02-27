# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for migrating random_walk parameters to samplerConfig"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="randomwalk_mode_to_sampler_config",
    resource_type=CoreResourceKinds.OPERATION,
    deprecated_fields=["mode", "grouping", "samplerType"],
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

    # Check if this is an operation with parameters that need migration
    config = data.get("config")
    if not isinstance(config, dict):
        return data

    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        return data

    # Check if mode field exists (indicator of old format)
    if "mode" not in parameters:
        return data

    # Extract the old fields
    mode = parameters.pop("mode", None)
    grouping = parameters.pop("grouping", None)
    sampler_type = parameters.pop("samplerType", None)

    # Create samplerConfig if any of the fields were present
    if mode is not None or grouping is not None or sampler_type is not None:
        sampler_config = {}
        if mode is not None:
            sampler_config["mode"] = mode
        if grouping is not None:
            sampler_config["grouping"] = grouping
        if sampler_type is not None:
            sampler_config["samplerType"] = sampler_type

        parameters["samplerConfig"] = sampler_config

    return data


# Made with Bob
