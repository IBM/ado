# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for SQLSampleStore.upgrade_entities() and upgrade_measurement_results()."""

import json
import uuid
from collections.abc import Callable

import sqlalchemy

from ado.core.samplestore.sql import SQLSampleStore
from ado.schema.entity import Entity
from ado.schema.result import (
    InvalidMeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)


def test_upgrade_entities_round_trips_and_returns_count(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """upgrade_entities() rewrites entity rows and returns the correct count."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(3)
    add_entities_to_sample_store(store, entities)

    entity_table = store._metadata.tables[store._tablename]
    target = entities[0]

    # Overwrite the stored representation with model_dump_json() WITHOUT
    # exclude_defaults=True.  This includes "measurement_results": []  and
    # every other default field. After upgrade the row should match the
    # canonical form produced by model_dump_json(exclude_defaults=True,
    # exclude={"measurement_results"}).
    pre_upgrade_json = target.model_dump_json()
    with store.engine.begin() as conn:
        conn.execute(
            entity_table.update()
            .where(entity_table.c.identifier == target.identifier)
            .values(representation=pre_upgrade_json)
        )

    # Verify pre-upgrade state contains default/excluded fields
    with store.engine.begin() as conn:
        row = conn.execute(
            sqlalchemy.select(entity_table.c.representation).where(
                entity_table.c.identifier == target.identifier
            )
        ).fetchone()
    pre = json.loads(row[0])
    # model_dump_json() without exclude_defaults includes measurement_results
    assert "measurement_results" in pre

    # Run upgrade
    count = store.upgrade_entities()
    assert count == 3

    # Verify the row now matches the canonical Pydantic serialisation
    with store.engine.begin() as conn:
        row = conn.execute(
            sqlalchemy.select(entity_table.c.representation).where(
                entity_table.c.identifier == target.identifier
            )
        ).fetchone()
    post = json.loads(row[0])
    expected = json.loads(
        target.model_dump_json(exclude_defaults=True, exclude={"measurement_results"})
    )
    assert post == expected
    # measurement_results excluded from the canonical form
    assert "measurement_results" not in post


def test_upgrade_entities_empty_store_returns_zero(
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """upgrade_entities() on an empty store returns 0."""
    store = random_sql_sample_store()
    count = store.upgrade_entities()
    assert count == 0


def _make_old_format_result(entity_identifier: str) -> dict:
    """Build the old (uncompressed) ValidMeasurementResult wire format.

    In the old format each ObservedPropertyValue contains the full
    experimentReference inside its property dict. The new compressed format
    stores experimentReference once at the top level.
    """
    exp_ref = {
        "experimentIdentifier": "benchmark_performance",
        "actuatorIdentifier": "replay",
    }
    measurement = {
        "value": 0.42,
        "property": {
            "targetProperty": {"identifier": "wallClockRuntime"},
            "experimentReference": exp_ref,  # OLD: redundant per measurement
        },
    }
    return {
        "uid": str(uuid.uuid4()),
        "entityIdentifier": entity_identifier,
        "measurements": [measurement],
        # OLD format: no top-level experimentReference
    }


def test_upgrade_measurement_results_compresses_old_format(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """upgrade_measurement_results() re-serializes old-format rows to compressed format."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(1)
    add_entities_to_sample_store(store, entities)
    entity = entities[0]

    # Insert an old-format result row directly via SQL.
    res_table = store._result_table
    old_data = _make_old_format_result(entity.identifier)
    with store.engine.begin() as conn:
        conn.execute(
            sqlalchemy.insert(res_table).values(
                uid=old_data["uid"],
                entity_id=entity.identifier,
                data=old_data,
            )
        )

    # Verify it is stored in old format (experimentReference inside each measurement's property)
    with store.engine.begin() as conn:
        row = conn.execute(
            sqlalchemy.select(res_table.c.data).where(
                res_table.c.uid == old_data["uid"]
            )
        ).fetchone()
    stored_before = row[0]
    assert "experimentReference" in stored_before["measurements"][0]["property"]
    assert "experimentReference" not in stored_before  # not at top level yet

    # Run upgrade
    count = store.upgrade_measurement_results()
    assert count == 1

    # Verify the row is now in the new compressed format
    with store.engine.begin() as conn:
        row = conn.execute(
            sqlalchemy.select(res_table.c.data).where(
                res_table.c.uid == old_data["uid"]
            )
        ).fetchone()
    stored_after = row[0]

    # New format: experimentReference at top level
    assert "experimentReference" in stored_after
    # New format: measurements should NOT contain redundant experimentReference
    assert "experimentReference" not in stored_after["measurements"][0]["property"]


def test_upgrade_measurement_results_empty_store_returns_zero(
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """upgrade_measurement_results() on an empty store returns 0."""
    store = random_sql_sample_store()
    count = store.upgrade_measurement_results()
    assert count == 0


def test_upgrade_measurement_results_preserves_invalid_results(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None],
        ValidMeasurementResult | InvalidMeasurementResult,
    ],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    """upgrade_measurement_results() correctly handles InvalidMeasurementResult rows."""
    store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(1)
    add_entities_to_sample_store(store, entities)
    entity = entities[0]

    result = random_ml_multi_cloud_benchmark_performance_measurement_results(
        entity, 1, MeasurementResultStateEnum.INVALID
    )
    assert isinstance(result, InvalidMeasurementResult)

    store.add_measurement_results([result], skip_relationship_to_request=True)

    count = store.upgrade_measurement_results()
    assert count == 1

    # Verify the row can still be deserialized as InvalidMeasurementResult
    res_table = store._result_table
    with store.engine.begin() as conn:
        row = conn.execute(
            sqlalchemy.select(res_table.c.data).where(res_table.c.uid == result.uid)
        ).fetchone()
    stored = row[0]
    assert "reason" in stored
    reloaded = InvalidMeasurementResult.model_validate(stored)
    assert reloaded.reason == result.reason
    assert reloaded.entityIdentifier == entity.identifier


def _make_legacy_entity_representation(
    entity_identifier: str,
    generatorid: str = "explicit_grid_sample_generator",
) -> str:
    """Return a JSON string in the old flat ``propertyValues`` entity format.

    The legacy format stored constitutive and observed property values in a
    single ``propertyValues`` list and all property definitions in a parallel
    ``properties`` list. Observed entries are distinguished by having a
    ``targetProperty`` key inside their ``property`` dict.
    """
    exp_ref = {
        "experimentIdentifier": "finetune-full-fsdp-v1.0.0",
        "actuatorIdentifier": "SFTTrainer",
    }
    return json.dumps(
        {
            "identifier": entity_identifier,
            "generatorid": generatorid,
            "propertyValues": [
                # Constitutive property value
                {
                    "value": "granite-13b-v2",
                    "property": {
                        "identifier": "model_name",
                        "propertyDomain": {
                            "values": ["granite-13b-v2"],
                            "variableType": "CATEGORICAL_VARIABLE_TYPE",
                        },
                    },
                },
                # Observed property values (embedded in old format)
                {
                    "value": 0.0,
                    "property": {
                        "targetProperty": {"identifier": "is_valid"},
                        "experimentReference": exp_ref,
                    },
                },
                {
                    "value": 42.5,
                    "property": {
                        "targetProperty": {"identifier": "train_runtime"},
                        "experimentReference": exp_ref,
                    },
                },
            ],
            "properties": [
                {
                    "identifier": "model_name",
                    "propertyDomain": {
                        "values": ["granite-13b-v2"],
                        "variableType": "CATEGORICAL_VARIABLE_TYPE",
                    },
                },
                {
                    "targetProperty": {"identifier": "is_valid"},
                    "experimentReference": exp_ref,
                },
                {
                    "targetProperty": {"identifier": "train_runtime"},
                    "experimentReference": exp_ref,
                },
            ],
        }
    )


def test_upgrade_entities_migrates_legacy_measurement_results(
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """upgrade_entities() extracts embedded measurements from legacy entity rows.

    When the stored entity representation uses the old ``propertyValues`` format,
    any observed property values it contains must be written to the result table
    so they are not silently discarded when the entity row is re-serialized in
    the new format.
    """
    store = random_sql_sample_store()
    entity_identifier = "model_name.granite-13b-v2"
    legacy_repr = _make_legacy_entity_representation(entity_identifier)

    # Insert the legacy entity row directly via SQL, bypassing the normal write path.
    entity_table = store._metadata.tables[store._tablename]
    with store.engine.begin() as conn:
        conn.execute(
            sqlalchemy.insert(entity_table).values(
                identifier=entity_identifier,
                representation=legacy_repr,
            )
        )

    # Confirm pre-upgrade state: legacy format in the entity table, no result rows.
    res_table = store._result_table
    with store.engine.begin() as conn:
        entity_row = conn.execute(
            sqlalchemy.select(entity_table.c.representation).where(
                entity_table.c.identifier == entity_identifier
            )
        ).fetchone()
        result_count_before = conn.execute(
            sqlalchemy.select(sqlalchemy.func.count()).select_from(res_table)
        ).scalar()

    assert "propertyValues" in json.loads(entity_row[0])
    assert result_count_before == 0

    # Run upgrade.
    count = store.upgrade_entities()
    assert count == 1

    # Entity row must now be in the new format (no ``propertyValues``).
    with store.engine.begin() as conn:
        entity_row = conn.execute(
            sqlalchemy.select(entity_table.c.representation).where(
                entity_table.c.identifier == entity_identifier
            )
        ).fetchone()
    upgraded = json.loads(entity_row[0])
    assert "propertyValues" not in upgraded
    assert "constitutive_property_values" in upgraded

    # The two observed property values must now live in the result table.
    with store.engine.begin() as conn:
        result_rows = conn.execute(
            sqlalchemy.select(res_table.c.data).where(
                res_table.c.entity_id == entity_identifier
            )
        ).fetchall()

    assert len(result_rows) == 1  # one ValidMeasurementResult (same experiment ref)
    result_data = result_rows[0][0]
    measured_targets = {
        m["property"]["targetProperty"]["identifier"]
        for m in result_data["measurements"]
    }
    assert measured_targets == {"is_valid", "train_runtime"}


def test_upgrade_entities_legacy_migration_is_idempotent(
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """Running upgrade_entities() twice does not duplicate result rows."""
    store = random_sql_sample_store()
    entity_identifier = "model_name.granite-13b-v2"
    legacy_repr = _make_legacy_entity_representation(entity_identifier)

    entity_table = store._metadata.tables[store._tablename]
    with store.engine.begin() as conn:
        conn.execute(
            sqlalchemy.insert(entity_table).values(
                identifier=entity_identifier,
                representation=legacy_repr,
            )
        )

    store.upgrade_entities()  # first run
    store.upgrade_entities()  # second run — must not duplicate

    res_table = store._result_table
    with store.engine.begin() as conn:
        result_count = conn.execute(
            sqlalchemy.select(sqlalchemy.func.count())
            .select_from(res_table)
            .where(res_table.c.entity_id == entity_identifier)
        ).scalar()

    assert result_count == 1
