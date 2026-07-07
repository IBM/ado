# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import TYPE_CHECKING

from ado.core import ADOResource, DiscoverySpaceResource, OperationResource
from ado.core.operation.config import DiscoveryOperationEnum
from tests.conftest import requires_sqlite_3_38

if TYPE_CHECKING:
    from ado.core.discoveryspace.stats import DiscoverySpaceStatistics
    from ado.metastore.sqlstore import SQLStore


@requires_sqlite_3_38
def test_get_space_metastore_stats_single_no_operations(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    sql_store: "SQLStore",
) -> None:
    """A space with no operations returns zero counts for all op fields."""
    space = random_space_resource_from_db()

    stats: DiscoverySpaceStatistics = sql_store.get_space_metastore_stats(
        space.identifier
    )

    assert stats.number_of_experiments == 1
    assert stats.number_of_operations == 0
    assert stats.number_of_explore_operations == 0


@requires_sqlite_3_38
def test_get_space_metastore_stats_single_with_explore_operation(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    create_resource_with_related_identifiers: Callable[[ADOResource, list[str]], None],
    sql_store: "SQLStore",
) -> None:
    """A space with one EXPLORE operation has correct counts."""
    space = random_space_resource_from_db()
    op = ml_multi_cloud_operation_resource(space_id=space.identifier)
    create_resource_with_related_identifiers(op, [space.identifier])

    stats: DiscoverySpaceStatistics = sql_store.get_space_metastore_stats(
        space.identifier
    )

    assert stats.number_of_experiments == 1
    assert stats.number_of_operations == 1
    assert stats.number_of_explore_operations == 1


@requires_sqlite_3_38
def test_get_space_metastore_stats_single_with_non_explore_operation(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    create_resource_with_related_identifiers: Callable[[ADOResource, list[str]], None],
    sql_store: "SQLStore",
) -> None:
    """A non-EXPLORE operation is counted in total but not in explore."""
    space = random_space_resource_from_db()
    op = ml_multi_cloud_operation_resource(space_id=space.identifier)
    op.operationType = DiscoveryOperationEnum.CHARACTERIZE
    create_resource_with_related_identifiers(op, [space.identifier])

    stats: DiscoverySpaceStatistics = sql_store.get_space_metastore_stats(
        space.identifier
    )

    assert stats.number_of_operations == 1
    assert stats.number_of_explore_operations == 0


@requires_sqlite_3_38
def test_get_space_metastore_stats_single_mixed_operations(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    create_resource_with_related_identifiers: Callable[[ADOResource, list[str]], None],
    sql_store: "SQLStore",
) -> None:
    """Mixed operation types: total and explore counts are both correct."""
    space = random_space_resource_from_db()

    explore_op = ml_multi_cloud_operation_resource(space_id=space.identifier)
    create_resource_with_related_identifiers(explore_op, [space.identifier])

    characterize_op = ml_multi_cloud_operation_resource(space_id=space.identifier)
    characterize_op.operationType = DiscoveryOperationEnum.CHARACTERIZE
    create_resource_with_related_identifiers(characterize_op, [space.identifier])

    stats: DiscoverySpaceStatistics = sql_store.get_space_metastore_stats(
        space.identifier
    )

    assert stats.number_of_operations == 2
    assert stats.number_of_explore_operations == 1


@requires_sqlite_3_38
def test_get_space_metastore_stats_multiple_spaces(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    create_resource_with_related_identifiers: Callable[[ADOResource, list[str]], None],
    sql_store: "SQLStore",
) -> None:
    """Multiple space IDs are returned in a dict keyed by space ID."""
    space_a = random_space_resource_from_db()
    space_b = random_space_resource_from_db()

    op_a = ml_multi_cloud_operation_resource(space_id=space_a.identifier)
    create_resource_with_related_identifiers(op_a, [space_a.identifier])
    # space_b has no operations

    stats = sql_store.get_space_metastore_stats(
        {space_a.identifier, space_b.identifier}
    )

    assert isinstance(stats, dict)
    assert space_a.identifier in stats
    assert space_b.identifier in stats

    assert stats[space_a.identifier].number_of_operations == 1
    assert stats[space_a.identifier].number_of_explore_operations == 1
    assert stats[space_b.identifier].number_of_operations == 0
    assert stats[space_b.identifier].number_of_explore_operations == 0
