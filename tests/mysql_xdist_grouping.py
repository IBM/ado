# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Classify collected pytest items and return MySQL ``xdist_group`` names."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Fixtures that start or require the Testcontainers MySQL instance.
MYSQL_CONTAINER_FIXTURES = frozenset(
    {
        "mysql_test_instance",
        "valid_ado_mysql_project_context",
        "valid_ado_mysql_context_yaml",
    }
)

# Parametrized fixtures whose ``mysql`` parameter requires MySQL.
MYSQL_PARAMETRIZED_FIXTURES = frozenset(
    {
        "valid_ado_project_context",
    }
)

# Number of MySQL ``xdist_group`` names to assign.
MYSQL_XDIST_GROUP_COUNT = 2


def requires_mysql_container(
    fixturenames: Iterable[str],
    params: Mapping[str, object] | None = None,
) -> bool:
    """Return whether a collected test item needs a MySQL testcontainer.

    Args:
        fixturenames: Fixture names known for the item at collection time.
        params: ``callspec.params`` for the item, if present.

    Returns:
        True when the item's fixtures or parameters require MySQL.
    """
    if MYSQL_CONTAINER_FIXTURES.intersection(fixturenames):
        return True
    if params is None:
        return False
    for fixture_name in MYSQL_PARAMETRIZED_FIXTURES:
        if params.get(fixture_name) == "mysql":
            return True
    return False


def mysql_xdist_group_name(
    nodeid: str,
    fixturenames: Iterable[str],
    params: Mapping[str, object] | None = None,
    group_count: int = MYSQL_XDIST_GROUP_COUNT,
) -> str | None:
    """Return an ``xdist_group`` name for a MySQL-using item.

    Args:
        nodeid: Pytest node id used to bucket the item stably.
        fixturenames: Fixture names known for the item at collection time.
        params: ``callspec.params`` for the item, if present.
        group_count: Number of concurrent MySQL xdist groups to use.

    Returns:
        A group name such as ``mysql_0``, or ``None``.
    """
    if group_count < 1:
        raise ValueError("group_count must be >= 1")
    if not requires_mysql_container(fixturenames, params):
        return None
    digest = hashlib.sha256(nodeid.encode()).hexdigest()
    bucket = int(digest, 16) % group_count
    return f"mysql_{bucket}"
