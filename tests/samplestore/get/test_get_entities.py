# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for SQLSampleStore.get_entities()."""

from collections.abc import Callable

import pytest

from ado.core.samplestore.sql import SQLSampleStore
from ado.schema.entity import Entity
from ado.schema.result import (
    MeasurementResult,
    MeasurementResultStateEnum,
)


def test_get_entities_returns_all_entities_without_measurements(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> None:
    """get_entities() with no arguments returns all entities, no measurements."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(4)
    for e in entities:
        e.measurement_results = []

    # Add measurements to DB too
    add_entities_to_sample_store(store, entities)
    for entity in entities:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        store.add_measurement_results([result], skip_relationship_to_request=True)

    result_entities = store.get_entities(require_measurements=False)

    assert len(result_entities) == 4
    # No measurements should be attached
    assert all(len(e.measurement_results) == 0 for e in result_entities)


def test_get_entities_single_string_identifier(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """get_entities('id') with a single string returns exactly that entity."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(3)
    add_entities_to_sample_store(store, entities)

    target_id = entities[1].identifier
    result = store.get_entities(target_id, require_measurements=False)

    assert len(result) == 1
    assert result[0].identifier == target_id


def test_get_entities_require_measurements_attaches_results(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> None:
    """get_entities(require_measurements=True) attaches measurements and updates tracking."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(3)
    for e in entities:
        e.measurement_results = []
    add_entities_to_sample_store(store, entities)

    for entity in entities:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        store.add_measurement_results([result], skip_relationship_to_request=True)

    result_entities = store.get_entities(require_measurements=True)

    assert len(result_entities) == 3
    assert all(len(e.measurement_results) == 1 for e in result_entities)

    # All ids should now be in the measurements-loaded tracking set
    entity_ids = {e.identifier for e in entities}
    assert entity_ids.issubset(store._entities_with_measurements_loaded)


def test_get_entities_require_measurements_second_call_no_requery(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> None:
    """Second get_entities(require_measurements=True) call does not re-attach measurements."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(2)
    for e in entities:
        e.measurement_results = []
    add_entities_to_sample_store(store, entities)

    for entity in entities:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        store.add_measurement_results([result], skip_relationship_to_request=True)

    _ = store.get_entities(require_measurements=True)
    loaded_before = set(store._entities_with_measurements_loaded)

    # Second call — all ids already in tracking set, no requery expected
    result_entities = store.get_entities(require_measurements=True)

    assert all(len(e.measurement_results) == 1 for e in result_entities)
    # Tracking set unchanged (no new ids added)
    assert store._entities_with_measurements_loaded == loaded_before


def test_get_entities_no_measurements_entity_still_marked_loaded(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Entity with no measurements in DB still appears in _entities_with_measurements_loaded."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(2)
    for e in entities:
        e.measurement_results = []
    add_entities_to_sample_store(store, entities)
    # Add NO measurements to the DB

    result_entities = store.get_entities(require_measurements=True)

    assert len(result_entities) == 2
    entity_ids = {e.identifier for e in entities}
    # Even with no measurements, entities should be marked as checked
    assert entity_ids.issubset(store._entities_with_measurements_loaded)


def test_get_entities_refresh_subset_evicts_and_refetches(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """get_entities({'id'}, refresh=True) evicts only requested ids and re-fetches them."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(3)
    add_entities_to_sample_store(store, entities)

    # Warm cache
    _ = store.get_entities(require_measurements=False)
    assert len(store._entities) == 3

    target_id = entities[0].identifier
    last_insert_id_before = store._last_insert_id

    # Refresh only the first entity
    result = store.get_entities({target_id}, refresh=True, require_measurements=False)

    assert len(result) == 1
    assert result[0].identifier == target_id
    # Other entities still in cache
    assert len(store._entities) == 3
    # _last_insert_id must not have been reset for a subset refresh
    assert store._last_insert_id == last_insert_id_before


def test_get_entities_refresh_all_evicts_everything(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """get_entities(refresh=True) with identifiers=None re-fetches everything."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(3)
    add_entities_to_sample_store(store, entities)

    # Warm cache and simulate a non-zero last_insert_id
    store._last_insert_id = 99
    _ = store.get_entities(require_measurements=False)

    # Full refresh
    result = store.get_entities(refresh=True, require_measurements=False)

    assert len(result) == 3
    # _last_insert_id should have been reset to 0 then remain 0 (no measurements)
    assert store._last_insert_id == 0


def test_get_entities_refresh_true_require_measurements(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> None:
    """get_entities(refresh=True, require_measurements=True) re-fetches entities and measurements."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(2)
    for e in entities:
        e.measurement_results = []
    add_entities_to_sample_store(store, entities)

    for entity in entities:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        store.add_measurement_results([result], skip_relationship_to_request=True)

    # First warm the cache
    _ = store.get_entities(require_measurements=True)
    assert {e.identifier for e in entities}.issubset(
        store._entities_with_measurements_loaded
    )

    # Full refresh with measurements
    result_entities = store.get_entities(refresh=True, require_measurements=True)

    assert len(result_entities) == 2
    assert all(len(e.measurement_results) == 1 for e in result_entities)
    assert {e.identifier for e in entities}.issubset(
        store._entities_with_measurements_loaded
    )


def test_get_entities_deprecated_entities_property(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> None:
    """Accessing .entities emits DeprecationWarning and returns entities with measurements."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(2)
    for e in entities:
        e.measurement_results = []
    add_entities_to_sample_store(store, entities)

    for entity in entities:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        store.add_measurement_results([result], skip_relationship_to_request=True)

    with pytest.warns(
        DeprecationWarning, match="SQLSampleStore.entities is deprecated"
    ):
        result_entities = store.entities

    assert len(result_entities) == 2
    assert all(len(e.measurement_results) == 1 for e in result_entities)
