# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator for adding UTC timezone to naive discovery space created timestamps"""

import datetime

from orchestrator.core.legacy.registry import legacy_migrator
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.utilities.dictionaries import get_nested_value


@legacy_migrator(
    identifier="discoveryspace_created_timezone_utc",
    resource_type=CoreResourceKinds.DISCOVERYSPACE,
    deprecated_field_paths=["created"],
    deprecated_from_version="1.0.0",
    removed_from_version="1.0.0",
    description="Adds UTC timezone information to naive discovery space created timestamps.",
)
def add_utc_timezone_to_created(data: dict) -> dict:
    """Add UTC timezone information to a naive top-level created timestamp.

    Args:
        data: The resource data dictionary.

    Returns:
        The migrated resource data dictionary.
    """
    created = get_nested_value(data, "created") if isinstance(data, dict) else None
    if not isinstance(created, str) or created.endswith("Z"):
        return data

    parsed = datetime.datetime.fromisoformat(created)
    if parsed.tzinfo is not None:
        return data

    data["created"] = f"{created}Z"
    return data
