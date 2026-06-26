# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for renaming operationName to operatorName in operations"""

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.utilities.dictionaries import get_nested_value, set_nested_value


@legacy_migrator(
    identifier="operation_operator_name_field_rename",
    resource_type=CoreResourceKinds.OPERATION,
    deprecated_field_paths=["config.operation.module.operationName"],
    deprecated_from_version="1.0.0",
    removed_from_version="1.0.0",
    description="Renames the deprecated 'operationName' field to 'operatorName' in operation module references.",
)
def rename_operation_name_field(data: dict) -> dict:
    """Rename config.operation.module.operationName to operatorName.

    Args:
        data: The resource data dictionary.

    Returns:
        The migrated resource data dictionary.
    """
    if not isinstance(data, dict):
        return data

    field_path = "config.operation.module.operationName"
    operator_name_path = "config.operation.module.operatorName"

    operation_name = get_nested_value(data, field_path)
    if operation_name is None:
        return data

    if get_nested_value(data, operator_name_path) is None:
        set_nested_value(data, operator_name_path, operation_name)

    module = get_nested_value(data, "config.operation.module")
    if isinstance(module, dict):
        module.pop("operationName", None)

    return data
