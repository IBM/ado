# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import json
import re
from collections.abc import Callable

import pytest
import sqlalchemy

from ado.core import DiscoverySpaceResource, OperationResource
from ado.core.resources import ADOResource, CoreResourceKinds
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.sqlstore import SQLStore
from ado.schema.entity import Entity
from ado.schema.request import MeasurementRequestStateEnum, ReplayedMeasurement


def test_resource_creation(
    resource_generator_from_file: tuple[CoreResourceKinds, str],
    create_resources: Callable[[list[ADOResource], SQLStore], None],
    sql_store: SQLStore,
    request: pytest.FixtureRequest,
) -> None:
    _resource_kind, generator = resource_generator_from_file
    resource = request.getfixturevalue(generator)()
    create_resources(resources=[resource], db=sql_store)
    assert sql_store.containsResourceWithIdentifier(identifier=resource.identifier)


def test_invalid_resource_creation(
    resource_generator_from_file: tuple[CoreResourceKinds, str],
    create_resources: Callable[[list[ADOResource], SQLStore], None],
    sql_store: SQLStore,
    request: pytest.FixtureRequest,
) -> None:
    _resource_kind, generator = resource_generator_from_file
    resource = request.getfixturevalue(generator)()
    with pytest.raises(
        ValueError,
        match=r"Cannot add resource, .*, that is not a subclass of ADOResource",
    ):
        create_resources(resources=[resource.config], db=sql_store)


def test_resource_cannot_be_created_twice(
    resource_generator_from_file: tuple[CoreResourceKinds, str],
    create_resources: Callable[[list[ADOResource], SQLStore], None],
    sql_store: SQLStore,
    request: pytest.FixtureRequest,
) -> None:
    _resource_kind, generator = resource_generator_from_file
    resource = request.getfixturevalue(generator)()
    create_resources(resources=[resource], db=sql_store)
    with pytest.raises(
        ValueError,
        match=f"Resource with id {re.escape(resource.identifier)} already present. "
        "Use updateResource if you want to overwrite it",
    ):
        create_resources([resource], db=sql_store)


def test_create_operation_with_related_space(
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    create_resource_with_related_identifiers: Callable[
        [ADOResource, list[str], SQLStore], None
    ],
    sql_store: SQLStore,
    get_single_resource_by_identifier: Callable[
        [str, CoreResourceKinds], ADOResource | None
    ],
    get_related_resource_identifiers_by_identifier: Callable[
        [str, CoreResourceKinds, str], dict[CoreResourceKinds, set[str]]
    ],
) -> None:
    quantity = 3

    operation = ml_multi_cloud_operation_resource()
    space_ids = [random_space_resource_from_db().identifier for _ in range(quantity)]
    create_resource_with_related_identifiers(
        resource=operation,
        related_identifiers=space_ids,
        db=sql_store,
    )

    assert (
        get_single_resource_by_identifier(
            operation.identifier, kind=CoreResourceKinds.OPERATION
        )
        is not None
    )
    related_resource_identifiers = get_related_resource_identifiers_by_identifier(
        operation.identifier,
        CoreResourceKinds.OPERATION,
        "parent",
    ).get(CoreResourceKinds.DISCOVERYSPACE, set())
    for space_id in space_ids:
        assert space_id in related_resource_identifiers


def test_exception_on_resource_with_related_identifier_if_related_id_does_not_exist(
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    create_resource_with_related_identifiers: Callable[
        [ADOResource, list[str], SQLStore], None
    ],
    sql_store: SQLStore,
) -> None:
    operation = ml_multi_cloud_operation_resource()
    nonexistent_related_id = "IDoNotExist"
    with pytest.raises(
        ValueError,
        match=f"Unknown resource identifier passed {re.escape(str([nonexistent_related_id]))}",
    ):
        create_resource_with_related_identifiers(
            resource=operation,
            related_identifiers=[nonexistent_related_id],
            db=sql_store,
        )


def test_add_entities_to_sample_store(
    random_entities: Callable[[int], list[Entity]],
    random_sql_sample_store: Callable[[], SQLSampleStore],
    add_entities_to_sample_store: Callable[[SQLSampleStore, list[Entity]], None],
) -> None:
    quantity = 3
    entities = random_entities(quantity=quantity)
    sample_store = random_sql_sample_store()
    add_entities_to_sample_store(sample_store, entities)
    entity_ids = {e.identifier for e in entities}
    assert entity_ids.issubset(sample_store.entity_identifiers())


def test_add_measurement_request_to_sample_store(
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    number_entities = 3
    measurements_per_result = 1
    request = random_ml_multi_cloud_benchmark_performance_measurement_requests(
        number_entities=number_entities, measurements_per_result=measurements_per_result
    )
    sample_store = random_sql_sample_store()
    request_db_id = sample_store.add_measurement_request(request=request)
    assert request_db_id is not None
    sample_store.add_measurement_results(
        results=request.measurements,
        skip_relationship_to_request=False,
        request_db_id=request_db_id,
    )
    # Verify results were persisted
    assert (
        sample_store.measurement_results_count_for_operation(
            operation_id=request.operation_id
        )
        == number_entities
    )


def test_add_external_entities(
    random_sql_sample_store: Callable[[], SQLSampleStore], entity: Entity
) -> None:

    sample_store = random_sql_sample_store()

    # Ensure the identifier of the entity is not in the DB
    assert len(sample_store.entity_identifiers().intersection({entity.identifier})) == 0

    # Add the entity and ensure it's there
    sample_store.add_external_entities([entity])
    assert len(sample_store.entity_identifiers().intersection({entity.identifier})) == 1

    #
    results = sample_store.get_entities(
        identifiers=entity.identifier, require_measurements=False
    )
    retrieved_entity = results[0] if results else None
    assert retrieved_entity is not None
    assert len(retrieved_entity.propertyValues) == len(entity.propertyValues)
    for i, property_value in enumerate(entity.propertyValues):
        assert (
            abs(property_value.value - retrieved_entity.propertyValues[i].value) < 1e-15
        )


def test_add_measurement_request_metadata_round_trip(
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """Metadata dict must survive a round-trip through add_measurement_request.

    Regression test for the Core-insert double-encoding bug: passing
    json.dumps(dict) into a JSON column causes the value to be stored as a
    JSON string rather than a JSON object, breaking metadata.* DB filters and
    making the stored value differ from what was written.
    """
    expected_metadata = {"k": "v", "nested": {"x": 1}}

    request = random_ml_multi_cloud_benchmark_performance_measurement_requests(
        number_entities=1, measurements_per_result=1
    )
    request.metadata = expected_metadata

    sample_store = random_sql_sample_store()
    request_db_id = sample_store.add_measurement_request(request=request)
    assert request_db_id is not None

    # Read the raw stored value directly from the DB to verify the JSON column
    # is stored as an object (not a double-encoded string).
    req_table = sample_store._request_table
    with sample_store.engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.select(req_table.c.metadata).where(
                req_table.c.uid == str(request_db_id)
            )
        ).fetchone()

    assert row is not None
    raw = row[0]
    # SQLAlchemy's JSON column returns a dict on SQLite and MySQL; if it ever
    # returns a string the column was double-encoded.
    if isinstance(raw, str):
        raw = json.loads(raw)
    assert isinstance(raw, dict), (
        f"metadata stored as {type(raw).__name__}, expected dict"
    )
    assert raw == expected_metadata


def test_contains_entity_with_identifier_true_and_false(
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
    random_sql_sample_store: Callable[[], SQLSampleStore],
) -> None:
    """containsEntityWithIdentifier must return True for known entities and False for unknown ones."""
    request = random_ml_multi_cloud_benchmark_performance_measurement_requests(
        number_entities=2, measurements_per_result=1
    )
    sample_store = random_sql_sample_store()
    # ReplayedMeasurement skips addEntities inside add_measurement_request, so
    # register entities explicitly — as the operator does in practice.
    sample_store.addEntities(list(request.entities))

    for entity in request.entities:
        assert sample_store.containsEntityWithIdentifier(entity.identifier) is True

    assert (
        sample_store.containsEntityWithIdentifier("entity-that-does-not-exist") is False
    )
