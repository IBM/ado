# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utility functions for working with nested dictionaries"""


def get_parent_dict_and_key(data: dict, path: str) -> tuple[dict | None, str | None]:
    """Navigate to a nested field path and return parent dict and field name

    This is a low-level helper used by set_nested_value, remove_nested_field,
    and has_nested_field. For reading values, use get_nested_value instead.

    Args:
        data: The data dictionary
        path: Dot-separated path (e.g., "config.specification.module.moduleType")

    Returns:
        Tuple of (parent_dict, field_name) or (None, None) if path doesn't exist

    Example:
        parent, field = get_parent_dict_and_key(data, "config.properties")
        if parent and field:
            parent.pop(field, None)
    """
    parts = path.split(".")
    current = data

    # Navigate to parent
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return None, None
        current = current[part]

    # Return parent dict and final field name
    if isinstance(current, dict):
        return current, parts[-1]

    return None, None


def get_nested_value(data: dict, path: str) -> object | None:
    """Get the value at a nested field path

    Args:
        data: The data dictionary
        path: Dot-separated path (e.g., "config.specification.module.moduleType")

    Returns:
        The value at the specified path, or None if path doesn't exist

    Example:
        value = get_nested_value(data, "config.moduleType")
        if value == "sample_store":
            # Do something
    """
    parent, field = get_parent_dict_and_key(data, path)
    if parent is not None and field is not None and field in parent:
        return parent[field]
    return None


def set_nested_value(data: dict, path: str, value: object) -> bool:
    """Set a value at a nested field path

    Args:
        data: The data dictionary
        path: Dot-separated path
        value: Value to set

    Returns:
        True if successful, False if path doesn't exist

    Example:
        data = {"config": {"specification": {"module": {}}}}
        set_nested_value(data, "config.specification.module.type", "sample_store")
        # data is now {"config": {"specification": {"module": {"type": "sample_store"}}}}
    """
    parent, field = get_parent_dict_and_key(data, path)
    if parent is not None and field is not None:
        parent[field] = value
        return True
    return False


def remove_nested_field(data: dict, path: str) -> bool:
    """Remove a field at a nested path

    Args:
        data: The data dictionary
        path: Dot-separated path

    Returns:
        True if field was removed, False if path doesn't exist

    Example:
        data = {"config": {"properties": ["a", "b"], "other": "value"}}
        remove_nested_field(data, "config.properties")
        # data is now {"config": {"other": "value"}}
    """
    parent, field = get_parent_dict_and_key(data, path)
    if parent is not None and field is not None and field in parent:
        parent.pop(field)
        return True
    return False


def has_nested_field(data: dict, path: str) -> bool:
    """Check if a nested field path exists

    Args:
        data: The data dictionary
        path: Dot-separated path

    Returns:
        True if the field exists, False otherwise

    Example:
        data = {"config": {"specification": {"module": {"moduleType": "test"}}}}
        has_nested_field(data, "config.specification.module.moduleType")  # Returns True
        has_nested_field(data, "config.nonexistent")  # Returns False
    """
    parent, field = get_parent_dict_and_key(data, path)
    return parent is not None and field is not None and field in parent


# Made with Bob
