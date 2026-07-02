# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for DataContainerStatistics and SQLResourceStore.get_datacontainer_stats."""

from collections.abc import Callable

from orchestrator.core import DataContainerResource
from orchestrator.core.datacontainer.resource import DataContainer, TabularData
from orchestrator.core.resources import ADOResource
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.utilities.location import SQLStoreConfiguration

# ---------------------------------------------------------------------------
# Integration tests for SQLResourceStore.get_datacontainer_stats
# ---------------------------------------------------------------------------


def test_get_datacontainer_stats_empty_set_returns_empty_dict(
    sql_store: SQLStore,
) -> None:
    """Passing an empty set returns {} immediately without any DB query."""
    result = sql_store.get_datacontainer_stats(set())
    assert result == {}


def test_get_datacontainer_stats_all_fields(
    sql_store: SQLStore,
    create_resources: Callable[[list[ADOResource]], None],
    testTabularDataString: TabularData,
    test_sample_store_location: SQLStoreConfiguration,
    random_identifier: Callable[[], str],
) -> None:
    """Container with tabularData, locationData, and data: all four counts match."""
    data = {"key1": {"nested": "value"}, "key2": [1, 2, 3]}
    dc = DataContainerResource(
        config=DataContainer(
            tabularData={"t1": testTabularDataString},
            locationData={"loc1": test_sample_store_location},
            data=data,
        )
    )
    create_resources([dc])

    result = sql_store.get_datacontainer_stats({dc.identifier})

    assert dc.identifier in result
    stats = result[dc.identifier]
    assert stats.number_of_tables == 1
    assert stats.number_of_locations == 1
    assert stats.number_of_key_values == 2
    assert stats.total_data_bytes > 0


def test_get_datacontainer_stats_partial_fields(
    sql_store: SQLStore,
    create_resources: Callable[[list[ADOResource]], None],
    random_identifier: Callable[[], str],
) -> None:
    """Container with only data set: tables and locations report 0."""
    dc = DataContainerResource(
        config=DataContainer(
            data={"only_key": "only_value"},
        )
    )
    create_resources([dc])

    result = sql_store.get_datacontainer_stats({dc.identifier})

    assert dc.identifier in result
    stats = result[dc.identifier]
    assert stats.number_of_tables == 0
    assert stats.number_of_locations == 0
    assert stats.number_of_key_values == 1
    assert stats.total_data_bytes > 0


def test_get_datacontainer_stats_total_data_bytes_excludes_metadata(
    sql_store: SQLStore,
    create_resources: Callable[[list[ADOResource]], None],
    random_identifier: Callable[[], str],
) -> None:
    """total_data_bytes excludes the metadata field from the byte count."""
    dc = DataContainerResource(
        config=DataContainer(
            data={"x": "y"},
        )
    )
    create_resources([dc])

    result = sql_store.get_datacontainer_stats({dc.identifier})
    stats = result[dc.identifier]

    # The reported bytes must exclude the metadata section.
    # Verify by computing the full config size vs reported bytes:
    # config JSON (all fields) > total_data_bytes because metadata is stripped.
    import json

    full_config_json = json.dumps(dc.config.model_dump())
    full_config_bytes = len(full_config_json)
    assert stats.total_data_bytes < full_config_bytes


def test_get_datacontainer_stats_unknown_id_returns_zeros(
    sql_store: SQLStore,
) -> None:
    """An ID not in the store is returned with all-zero stats."""
    unknown_id = "datacontainer-does-not-exist"
    result = sql_store.get_datacontainer_stats({unknown_id})

    assert unknown_id in result
    stats = result[unknown_id]
    assert stats.number_of_tables == 0
    assert stats.number_of_locations == 0
    assert stats.number_of_key_values == 0
    assert stats.total_data_bytes == 0
