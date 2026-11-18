# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Limit concurrent MySQL testcontainers under pytest-xdist.

Each xdist worker is a separate process, so a session-scoped MySQL container
fixture starts one container per worker that needs it. Marking MySQL-using
tests with a small set of ``xdist_group`` names keeps container count bounded
when pytest is run with ``--dist loadgroup``.
"""

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

# Parametrized fixtures whose ``mysql`` value pulls in the MySQL container
# (possibly via ``request.getfixturevalue``, which is not always visible in
# ``fixturenames`` at collection time).
MYSQL_PARAMETRIZED_FIXTURES = frozenset(
    {
        "valid_ado_project_context",
    }
)

# Concurrent MySQL containers under ``-n auto --dist loadgroup``.
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
        True if the item should be placed in a MySQL xdist group.
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
    """Return an ``xdist_group`` name for a MySQL-using item, or None.

    Args:
        nodeid: Pytest node id used to bucket the item stably.
        fixturenames: Fixture names known for the item at collection time.
        params: ``callspec.params`` for the item, if present.
        group_count: Number of concurrent MySQL xdist groups to use.

    Returns:
        A group name such as ``mysql_0``, or None if the item does not need
        MySQL.
    """
    if group_count < 1:
        raise ValueError("group_count must be >= 1")
    if not requires_mysql_container(fixturenames, params):
        return None
    digest = hashlib.sha256(nodeid.encode()).hexdigest()
    bucket = int(digest, 16) % group_count
    return f"mysql_{bucket}"
