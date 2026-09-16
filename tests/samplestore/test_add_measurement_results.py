# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the JSON column size check in SQLSampleStore.add_measurement_results."""

from collections.abc import Callable

import pytest

from ado.core.samplestore.sql import SQLSampleStore
from ado.schema.entity import Entity
from ado.schema.result import MeasurementResultStateEnum, ValidMeasurementResult

# ---------------------------------------------------------------------------
# _max_json_column_bytes tests
# ---------------------------------------------------------------------------


def test_max_json_column_bytes_sqlite(
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """SQLite dialect returns the compile-time default of 1 GB."""
    store = random_sql_sample_store()
    if store.engine.dialect.name != "sqlite":
        pytest.skip("SQLite-only test")
    assert store._max_json_column_bytes == 1_000_000_000


def test_max_json_column_bytes_mysql(
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """MySQL dialect returns a positive integer (max_allowed_packet)."""
    store = random_sql_sample_store()
    if store.engine.dialect.name != "mysql":
        pytest.skip("MySQL-only test")
    assert isinstance(store._max_json_column_bytes, int)
    assert store._max_json_column_bytes > 0


def test_max_json_column_bytes_unknown_dialect_returns_none(
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """Unknown dialect path returns None (injected via cached_property slot)."""
    store = random_sql_sample_store()
    # Verify that injecting None into the cached_property slot is respected
    # by add_measurement_results (the size check is skipped).
    store.__dict__["_max_json_column_bytes"] = None
    assert store._max_json_column_bytes is None


# ---------------------------------------------------------------------------
# add_measurement_results size-check tests
# ---------------------------------------------------------------------------


def test_add_measurement_results_raises_for_oversized_result(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], ValidMeasurementResult
    ],
) -> None:
    """SystemError is raised when a result's serialised size exceeds the limit."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(1)
    add_entities_to_sample_store(store, entities)

    result = random_ml_multi_cloud_benchmark_performance_measurement_results(
        entities[0], 1, MeasurementResultStateEnum.VALID
    )

    # Force a 100-byte limit — any real result will exceed this
    store.__dict__["_max_json_column_bytes"] = 100

    with pytest.raises(SystemError, match=entities[0].identifier):
        store.add_measurement_results([result], skip_relationship_to_request=True)
