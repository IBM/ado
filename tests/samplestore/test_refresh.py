# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for SQL Sample Store refresh functionality."""

import copy
from collections.abc import Callable

from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.schema.entity import Entity
from orchestrator.schema.result import (
    MeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)


def test_fetch_entities_all(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test _fetch_entities() fetches all entities when entity_ids=None."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(5)
    add_entities_to_sample_store(store, entities)

    # Fetch all entities
    fetched = store._fetch_entities(entity_ids=None)

    assert len(fetched) == 5
    for entity in entities:
        assert entity.identifier in fetched
        assert fetched[entity.identifier].identifier == entity.identifier


def test_fetch_entities_filtered(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test _fetch_entities() fetches only specified entities."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(5)
    add_entities_to_sample_store(store, entities)

    # Fetch only 2 specific entities
    target_ids = {entities[0].identifier, entities[2].identifier}
    fetched = store._fetch_entities(entity_ids=target_ids)

    assert len(fetched) == 2
    assert entities[0].identifier in fetched
    assert entities[2].identifier in fetched
    assert entities[1].identifier not in fetched


def test_fetch_entities_empty_set_fetches_all(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test _fetch_entities() treats empty set same as None."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(3)
    add_entities_to_sample_store(store, entities)

    # Empty set should fetch all entities
    fetched = store._fetch_entities(entity_ids=set())

    assert len(fetched) == 3
    for entity in entities:
        assert entity.identifier in fetched


def test_fetch_entities_nonexistent(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test _fetch_entities() returns empty dict for nonexistent IDs."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(2)
    add_entities_to_sample_store(store, entities)

    # Try to fetch nonexistent entities
    fetched = store._fetch_entities(entity_ids={"nonexistent1", "nonexistent2"})

    assert len(fetched) == 0


def test_fetch_measurement_results_all(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test _fetch_measurement_results() fetches all results when min_insert_id=0."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(3)

    # Add measurements to entities
    measurement_results = []
    for entity in entities:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        assert isinstance(result, ValidMeasurementResult)
        entity.add_measurement_result(result)
        measurement_results.append(result)

    add_entities_to_sample_store(store, entities)
    store.add_measurement_results(
        measurement_results, skip_relationship_to_request=True
    )

    # Fetch all measurement results
    results_by_entity, max_insert_id = store._fetch_measurement_results(min_insert_id=0)

    assert len(results_by_entity) == 3
    assert max_insert_id > 0
    for entity in entities:
        assert entity.identifier in results_by_entity
        assert len(results_by_entity[entity.identifier]) == 1


def test_fetch_measurement_results_incremental(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test _fetch_measurement_results() fetches only new results."""
    store = random_sql_sample_store()

    # Add first batch of entities with measurements
    entities_batch1 = random_ml_multi_cloud_benchmark_performance_entities(2)
    measurement_results_batch1 = []
    for entity in entities_batch1:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        assert isinstance(result, ValidMeasurementResult)
        entity.add_measurement_result(result)
        measurement_results_batch1.append(result)
    add_entities_to_sample_store(store, entities_batch1)
    store.add_measurement_results(
        measurement_results_batch1, skip_relationship_to_request=True
    )

    # Get max insert_id from first batch
    _, max_insert_id_batch1 = store._fetch_measurement_results(min_insert_id=0)

    # Add second batch of entities with measurements
    entities_batch2 = random_ml_multi_cloud_benchmark_performance_entities(2)
    measurement_results_batch2 = []
    for entity in entities_batch2:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        assert isinstance(result, ValidMeasurementResult)
        entity.add_measurement_result(result)
        measurement_results_batch2.append(result)
    add_entities_to_sample_store(store, entities_batch2)
    store.add_measurement_results(
        measurement_results_batch2, skip_relationship_to_request=True
    )

    # Fetch only new results
    results_by_entity, max_insert_id_batch2 = store._fetch_measurement_results(
        min_insert_id=max_insert_id_batch1
    )

    # Should only get batch2 results
    assert len(results_by_entity) == 2
    assert max_insert_id_batch2 > max_insert_id_batch1
    for entity in entities_batch2:
        assert entity.identifier in results_by_entity


def test_refresh_initial_load(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test refresh() with force_fetch_all_entities=True for initial load."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(5)

    # Add measurements to entities
    measurement_results = []
    for entity in entities:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        assert isinstance(result, ValidMeasurementResult)
        entity.add_measurement_result(result)
        measurement_results.append(result)

    add_entities_to_sample_store(store, entities)
    store.add_measurement_results(
        measurement_results, skip_relationship_to_request=True
    )

    # Clear the cache to simulate fresh load
    store._entities = {}

    # Perform initial load via refresh
    new_entities_count, new_measurements_count = store.refresh(
        force_fetch_all_entities=True
    )

    assert new_entities_count == 5
    assert new_measurements_count == 5
    assert len(store._entities) == 5
    assert store._last_insert_id > 0


def test_refresh_incremental(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test refresh() fetches only new data in incremental mode."""
    store = random_sql_sample_store()

    # Initial load
    entities_batch1 = random_ml_multi_cloud_benchmark_performance_entities(1)
    measurement_results_batch1 = []
    for entity in entities_batch1:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        assert isinstance(result, ValidMeasurementResult)

        # These entities are from a replay experiment and they likely have
        # results already in them - we override them with the new result
        # that we generate
        entity.measurement_results = [result]
        measurement_results_batch1.append(result)

    store.addEntities(entities=entities_batch1)
    store.add_measurement_results(
        results=measurement_results_batch1, skip_relationship_to_request=True
    )

    # Trigger initial load
    _ = store.refresh(force_fetch_all_entities=True)

    # Save copies of the state after the first load
    initial_last_insert_id = int(store._last_insert_id)
    frozen_entities = copy.deepcopy(store._entities)

    # Add new entities with measurements
    entities_batch2 = random_ml_multi_cloud_benchmark_performance_entities(2)
    measurement_results_batch2 = []
    for entity in entities_batch2:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        assert isinstance(result, ValidMeasurementResult)

        # These entities are from a replay experiment and they likely have
        # results already in them - we override them with the new result
        # that we generate
        entity.measurement_results = [result]
        measurement_results_batch2.append(result)

    store.addEntities(entities=entities_batch2)
    store.add_measurement_results(
        results=measurement_results_batch2, skip_relationship_to_request=True
    )

    # The fixture might give us back entities with the same
    # identifier to the one we already had
    batch_1_entity_identifiers = {e.identifier for e in entities_batch1}
    batch_2_entity_identifiers = {e.identifier for e in entities_batch2}
    expected_new_entities = len(
        batch_2_entity_identifiers.difference(batch_1_entity_identifiers)
    )

    # Restore the state of the sample store to the previous state.
    # This simulates another process having updated it.
    store._entities = copy.deepcopy(frozen_entities)
    store._last_insert_id = int(initial_last_insert_id)

    # Perform incremental refresh
    new_entities_count, new_measurements_count = store.refresh()

    assert new_entities_count == expected_new_entities  # Only new entities
    assert new_measurements_count == 2  # Only new measurements
    assert len(store._entities) == 1 + expected_new_entities  # Total entities
    assert store._last_insert_id > initial_last_insert_id


def test_refresh_no_new_data(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test refresh() returns (0, 0) when no new data."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(2)
    add_entities_to_sample_store(store, entities)

    # Trigger initial load
    _ = store.entities

    # Refresh without adding new data
    new_entities_count, new_measurements_count = store.refresh()

    assert new_entities_count == 0
    assert new_measurements_count == 0


def test_refresh_adds_measurements_to_existing_entities(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """Test refresh() adds new measurements to existing entities."""
    store = random_sql_sample_store()

    # Add entities without measurements
    entities = random_ml_multi_cloud_benchmark_performance_entities(2)

    # These are replayed measurements and they already contain results
    # replace them with empty lists before adding them to the sample store
    for e in entities:
        e.measurement_results = []
    add_entities_to_sample_store(store, entities)

    # Trigger initial load
    initial_entities = store.entities
    assert all(len(e.measurement_results) == 0 for e in initial_entities)

    # Add measurements to existing entities directly in DB
    for entity in entities:
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, MeasurementResultStateEnum.VALID
        )
        store.add_measurement_results([result], skip_relationship_to_request=True)

    # Refresh to get new measurements
    new_entities_count, new_measurements_count = store.refresh()

    assert new_entities_count == 0  # No new entities
    assert new_measurements_count == 2  # New measurements added

    # Verify measurements were added to existing entities
    refreshed_entities = store.entities
    assert all(len(e.measurement_results) == 1 for e in refreshed_entities)
