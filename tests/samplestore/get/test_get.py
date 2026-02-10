# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import random
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from orchestrator.core import ADOResource, CoreResourceKinds
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from orchestrator.schema.result import (
    InvalidMeasurementResult,
    MeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)

if TYPE_CHECKING:
    from orchestrator.core.operation.config import (
        DiscoveryOperationResourceConfiguration,
    )

sqlite3_version = sqlite3.sqlite_version_info


def test_get_single_resource_by_id(
    resource_generator_from_db: tuple[CoreResourceKinds, str],
    get_single_resource_by_identifier: Callable[
        [str, CoreResourceKinds], ADOResource | None
    ],
    request: pytest.FixtureRequest,
) -> None:
    resource_kind, generator = resource_generator_from_db
    resource = request.getfixturevalue(generator)()
    assert resource.identifier is not None

    db_resource = get_single_resource_by_identifier(
        identifier=resource.identifier, kind=resource_kind
    )
    assert db_resource is not None


# AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
# ref: https://sqlite.org/json1.html#jptr
@pytest.mark.skipif(
    sqlite3_version < (3, 38, 0), reason="SQLite version 3.38.0 or higher is required"
)
def test_get_all_resources_of_kind(
    resource_generator_from_db: tuple[CoreResourceKinds, str],
    get_resource_identifiers_by_resource_kind: Callable[[str], pd.DataFrame],
    request: pytest.FixtureRequest,
) -> None:
    resource_kind, generator = resource_generator_from_db
    quantity = 3
    for _ in range(quantity):
        request.getfixturevalue(generator)()

    resources = get_resource_identifiers_by_resource_kind(kind=resource_kind.value)
    assert resources.shape[0], quantity


def test_cannot_get_resources_of_kind_for_wrong_kind(
    get_resource_identifiers_by_resource_kind: Callable[[str], pd.DataFrame],
) -> None:
    with pytest.raises(ValueError, match="Unknown kind specified: IDoNotExist"):
        get_resource_identifiers_by_resource_kind(kind="IDoNotExist")


def test_get_multiple_resources_by_id(
    resource_generator_from_db: tuple[CoreResourceKinds, str],
    get_multiple_resources_by_identifier: Callable[[list[str]], dict[str, ADOResource]],
    request: pytest.FixtureRequest,
) -> None:
    resource_kind, generator = resource_generator_from_db
    quantity = 3
    resource_ids = [
        request.getfixturevalue(generator)().identifier for _ in range(quantity)
    ]
    assert len(resource_ids), quantity
    resources = get_multiple_resources_by_identifier(identifiers=resource_ids)
    assert len(resources), quantity
    for resource in resources.values():
        assert resource.kind == resource_kind


def test_count_measurement_requests_and_results(
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_identifier: Callable[[], str],
) -> None:
    assert ml_multi_cloud_benchmark_performance_experiment is not None

    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, _, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=measurements_per_result,
        operation_id=operation_id,
    )

    assert (
        sample_store.measurement_requests_count_for_operation(operation_id=operation_id)
        == number_requests
    )
    assert (
        sample_store.measurement_results_count_for_operation(operation_id=operation_id)
        == number_requests * number_entities
    )


def test_operation_entity_statistics_all_valid(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test operation_entity_statistics with all valid measurement results."""
    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, _requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    # The random results might include the same entity multiple times.
    # To be sure of how many entities we should expect to see, we need
    # to check all the measurements of all the requests
    expected_entities = {
        result.entityIdentifier
        for request in _requests
        for result in request.measurements
    }

    # Get statistics
    stats = sample_store.operation_entity_statistics(operation_id=operation_id)

    # Verify counts - all entities should have all successful measurements
    assert stats["total_entities"] == len(expected_entities)
    # All entities from this operation should have all successful measurements
    # (since fixture creates all valid results)
    assert stats["entities_with_all_successful_measurements"] == len(expected_entities)
    assert stats["entities_with_at_least_one_successful_measurement"] == len(
        expected_entities
    )

    # Verify logical consistency
    assert (
        stats["entities_with_all_successful_measurements"]
        <= stats["entities_with_at_least_one_successful_measurement"]
    )
    assert (
        stats["entities_with_at_least_one_successful_measurement"]
        <= stats["total_entities"]
    )


def test_operation_entity_statistics_mixed_valid_invalid(
    random_identifier: Callable[[], str],
    ml_multi_cloud_sample_store: SQLSampleStore,
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
    valid_ado_project_context: ProjectContext,
    ml_multi_cloud_operation_configuration: "DiscoveryOperationResourceConfiguration",
) -> None:
    """Test operation_entity_statistics with mixed valid and invalid results."""
    from orchestrator.core import OperationResource
    from orchestrator.core.operation.config import DiscoveryOperationEnum
    from orchestrator.metastore.sqlstore import SQLResourceStore
    from orchestrator.schema.reference import ExperimentReference

    number_entities = 3
    measurements_per_result = 2
    operation_id = random_identifier()
    sample_store = ml_multi_cloud_sample_store

    # Create operation in metastore
    sql = SQLResourceStore(project_context=valid_ado_project_context, ensureExists=True)
    sql.addResourceWithRelationships(
        OperationResource(
            identifier=operation_id,
            config=ml_multi_cloud_operation_configuration,
            operationType=DiscoveryOperationEnum.SEARCH,
            operatorIdentifier="test-operator",
        ),
        relatedIdentifiers=ml_multi_cloud_operation_configuration.spaces,
    )

    # Create entities once - these will be reused across both requests
    entities = random_ml_multi_cloud_benchmark_performance_entities(number_entities)

    # Create mixed scenarios from the start:
    # Entity 0: 2 valid results -> all successful
    # Entity 1: 1 valid + 1 invalid result -> partially successful
    # Entity 2: 2 invalid results -> no successful

    # Request 0: All entities with valid results
    measurements_req0 = [
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity=entities[0],
            measurements_per_result=measurements_per_result,
            status=MeasurementResultStateEnum.VALID,
        ),
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity=entities[1],
            measurements_per_result=measurements_per_result,
            status=MeasurementResultStateEnum.VALID,
        ),
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity=entities[2],
            measurements_per_result=measurements_per_result,
            status=MeasurementResultStateEnum.INVALID,
        ),
    ]

    request0 = ReplayedMeasurement(
        operation_id=operation_id,
        requestIndex=0,
        experimentReference=ExperimentReference(
            experimentIdentifier="benchmark_performance",
            actuatorIdentifier="replay",
        ),
        entities=entities,
        requestid=random_identifier(),
        status=MeasurementRequestStateEnum.SUCCESS,
        measurements=tuple(measurements_req0),
    )
    request_id0 = sample_store.add_measurement_request(request=request0)
    sample_store.add_measurement_results(
        results=measurements_req0,
        skip_relationship_to_request=False,
        request_db_id=request_id0,
    )

    # Request 1: Entity 0 valid, Entity 1 invalid, Entity 2 invalid
    measurements_req1 = [
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity=entities[0],
            measurements_per_result=measurements_per_result,
            status=MeasurementResultStateEnum.VALID,
        ),
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity=entities[1],
            measurements_per_result=measurements_per_result,
            status=MeasurementResultStateEnum.INVALID,
        ),
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity=entities[2],
            measurements_per_result=measurements_per_result,
            status=MeasurementResultStateEnum.INVALID,
        ),
    ]

    request1 = ReplayedMeasurement(
        operation_id=operation_id,
        requestIndex=1,
        experimentReference=ExperimentReference(
            experimentIdentifier="benchmark_performance",
            actuatorIdentifier="replay",
        ),
        entities=entities,
        requestid=random_identifier(),
        status=MeasurementRequestStateEnum.SUCCESS,
        measurements=tuple(measurements_req1),
    )
    request_id1 = sample_store.add_measurement_request(request=request1)
    sample_store.add_measurement_results(
        results=measurements_req1,
        skip_relationship_to_request=False,
        request_db_id=request_id1,
    )

    # Get statistics
    stats = sample_store.operation_entity_statistics(operation_id=operation_id)

    # Verify counts
    assert stats["total_entities"] == number_entities
    # Entity 0: all 2 valid -> entities_with_all_successful_measurements
    assert stats["entities_with_all_successful_measurements"] == 1
    # Entities 0 and 1 have at least one successful measurement
    assert stats["entities_with_at_least_one_successful_measurement"] == 2


def test_measurement_results_for_operation(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:

    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()
    results = []

    sample_store, requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    # We return requests sorted by requestIndex
    requests = sorted(requests, key=lambda r: r.requestIndex)
    for r in requests:
        results.extend(r.measurements)

    retrieved_results = sample_store.measurement_results_for_operation(
        operation_id=operation_id
    )

    # Check all the measurements are there
    assert len(retrieved_results) == len(results)

    for i, result in enumerate(results):
        assert result.__class__.__name__ == retrieved_results[i].__class__.__name__
        assert result.entityIdentifier == retrieved_results[i].entityIdentifier
        assert result.uid == retrieved_results[i].uid

        if isinstance(result, InvalidMeasurementResult):
            assert result.reason == retrieved_results[i].reason
            continue

        assert len(result.measurements) == len(retrieved_results[i].measurements)
        for j, measurement in enumerate(result.measurements):
            assert (
                abs(measurement.value - retrieved_results[i].measurements[j].value)
                < 1e-15
            )


def test_measurement_requests_for_operation(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:

    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()
    results = []

    sample_store, requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    # We return requests sorted by requestIndex
    requests = sorted(requests, key=lambda r: r.requestIndex)
    for r in requests:
        results.extend(r.measurements)

    retrieved_requests = sample_store.measurement_requests_for_operation(
        operation_id=operation_id
    )

    # Check all the measurement requests are there
    assert len(retrieved_requests) == len(requests)

    for i in range(len(requests)):
        # Check all the measurement results are there
        assert len(requests[i].measurements) == len(retrieved_requests[i].measurements)

        for j in range(len(requests[i].measurements)):
            # Check the values are correct
            if isinstance(requests[i].measurements[j], ValidMeasurementResult):

                assert (
                    requests[i].measurements[j].uid
                    == retrieved_requests[i].measurements[j].uid
                )

                assert len(requests[i].measurements[j].measurements) == len(
                    retrieved_requests[i].measurements[j].measurements
                )
                for k in range(len(requests[i].measurements[j].measurements)):
                    assert (
                        abs(
                            requests[i].measurements[j].measurements[k].value
                            - retrieved_requests[i]
                            .measurements[j]
                            .measurements[k]
                            .value
                        )
                        < 1e-15
                    )
            else:
                assert isinstance(
                    retrieved_requests[i].measurements[j], InvalidMeasurementResult
                )


def test_measurement_request_by_id(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:

    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    to_be_found: MeasurementRequest = random.choice(requests)
    result_from_db = sample_store.measurement_request_by_id(
        measurement_request_id=to_be_found.requestid
    )

    assert result_from_db is not None
    assert len(to_be_found.measurements) == len(result_from_db.measurements)

    for i in range(len(to_be_found.measurements)):
        # Check the values are correct
        if isinstance(to_be_found.measurements[i], ValidMeasurementResult):

            assert len(to_be_found.measurements[i].measurements) == len(
                result_from_db.measurements[i].measurements
            )
            for j in range(len(to_be_found.measurements[i].measurements)):
                assert (
                    abs(
                        to_be_found.measurements[i].measurements[j].value
                        - result_from_db.measurements[i].measurements[j].value
                    )
                    < 1e-15
                )
        else:
            assert isinstance(result_from_db.measurements[i], InvalidMeasurementResult)


def test_experiments_in_operation(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
) -> None:

    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, _requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    retrieved_experiment_references = sample_store.experiments_in_operation(
        operation_id=operation_id
    )
    assert len(retrieved_experiment_references) == 1
    assert (
        retrieved_experiment_references[0]
        == ml_multi_cloud_benchmark_performance_experiment
    )


def test_entity_identifiers_in_operation(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:

    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    entity_ids = set()
    for r in requests:
        entity_ids = entity_ids.union({e.identifier for e in r.entities})

    retrieved_entity_ids = sample_store.entity_identifiers_in_operation(
        operation_id=operation_id
    )
    assert len(entity_ids) == len(retrieved_entity_ids)
    assert len(entity_ids.intersection(retrieved_entity_ids)) == len(entity_ids)


def test_entity_identifiers_in_sample_store(
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:

    expected_identifiers = [e.identifier for e in ml_multi_cloud_sample_store.entities]
    retrieved_identifiers = ml_multi_cloud_sample_store.entity_identifiers()

    assert len(expected_identifiers) == len(retrieved_identifiers)
    assert len(set(expected_identifiers).intersection(retrieved_identifiers)) == len(
        retrieved_identifiers
    )


def test_entity_results_keep_uids(
    entity: Entity, ml_multi_cloud_sample_store: SQLSampleStore
) -> None:

    ml_multi_cloud_sample_store.add_external_entities([entity])
    retrieved_entity = ml_multi_cloud_sample_store.entityWithIdentifier(
        entityIdentifier=entity.identifier
    )

    assert len(entity.measurement_results) == len(retrieved_entity.measurement_results)
    for i in range(len(retrieved_entity.measurement_results)):
        assert (
            entity.measurement_results[i].uid
            == retrieved_entity.measurement_results[i].uid
        )


def test_float_precision_errors_when_retrieving_results(
    ml_multi_cloud_sample_store: SQLSampleStore,
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
) -> None:

    measurement_request = (
        random_ml_multi_cloud_benchmark_performance_measurement_requests(
            number_entities=1, measurements_per_result=1
        )
    )
    request_db_id = ml_multi_cloud_sample_store.add_measurement_request(
        measurement_request
    )
    ml_multi_cloud_sample_store.add_measurement_results(
        results=measurement_request.measurements,
        skip_relationship_to_request=False,
        request_db_id=request_db_id,
    )
    assert request_db_id is not None

    max_retries = 100
    errors_found = False
    for _ in range(max_retries):
        retrieved_request: MeasurementRequest = (
            ml_multi_cloud_sample_store.measurement_request_by_id(
                measurement_request_id=measurement_request.requestid
            )
        )
        if (
            retrieved_request.measurements[0].measurements[0].value
            != measurement_request.measurements[0].measurements[0].value
        ):
            float_inconsistency = abs(
                retrieved_request.measurements[0].measurements[0].value
                - measurement_request.measurements[0].measurements[0].value
            )
            assert (
                float_inconsistency < 1e-15
            ), f"The floats had an error bigger than 1e-15 (was {float_inconsistency}"
            errors_found = True
            break

    if not errors_found:
        pytest.xfail("No float inconsistency errors were spotted")
    else:
        assert errors_found


def test_entities_by_identifiers_empty_input(
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Test entities_with_identifiers with empty input returns empty list."""
    result = ml_multi_cloud_sample_store.entities_with_identifiers([])
    assert result == []

    result = ml_multi_cloud_sample_store.entities_with_identifiers(set())
    assert result == []


def test_entities_by_identifiers_list_input(
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Test entities_with_identifiers accepts list input."""
    # Get some entity identifiers from the store
    all_entities = ml_multi_cloud_sample_store.entities
    assert len(all_entities) > 0

    # Test with list input
    entity_ids_list = [all_entities[0].identifier, all_entities[1].identifier]
    result = ml_multi_cloud_sample_store.entities_with_identifiers(entity_ids_list)

    assert len(result) == 2
    assert {e.identifier for e in result} == set(entity_ids_list)


def test_entities_by_identifiers_set_input(
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Test entities_with_identifiers accepts set input."""
    # Get some entity identifiers from the store
    all_entities = ml_multi_cloud_sample_store.entities
    assert len(all_entities) > 0

    # Test with set input
    entity_ids_set = {all_entities[0].identifier, all_entities[1].identifier}
    result = ml_multi_cloud_sample_store.entities_with_identifiers(entity_ids_set)

    assert len(result) == 2
    assert {e.identifier for e in result} == entity_ids_set


def test_entities_by_identifiers_subset(
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Test entities_with_identifiers returns only requested entities."""
    # Get all entities from the store
    all_entities = ml_multi_cloud_sample_store.entities
    assert len(all_entities) >= 3

    # Request only a subset
    requested_ids = {all_entities[0].identifier, all_entities[2].identifier}
    result = ml_multi_cloud_sample_store.entities_with_identifiers(requested_ids)

    assert len(result) == 2
    assert {e.identifier for e in result} == requested_ids

    # Verify we got the correct entities
    result_ids = {e.identifier for e in result}
    expected_entity_ids = {
        e.identifier for e in all_entities if e.identifier in requested_ids
    }
    assert len(result) == len(expected_entity_ids)
    assert result_ids == expected_entity_ids
    for entity in result:
        assert entity.identifier in requested_ids


def test_entities_by_identifiers_nonexistent_entities(
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Test entities_with_identifiers with non-existent entity identifiers."""
    # Request entities that don't exist
    nonexistent_ids = {"nonexistent_id_1", "nonexistent_id_2"}
    result = ml_multi_cloud_sample_store.entities_with_identifiers(nonexistent_ids)

    # Should return empty list, not raise an error
    assert result == []


def test_entities_by_identifiers_mixed_existing_nonexistent(
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Test entities_with_identifiers with mix of existing and non-existent identifiers."""
    all_entities = ml_multi_cloud_sample_store.entities
    assert len(all_entities) > 0

    # Mix of existing and non-existent
    mixed_ids = {
        all_entities[0].identifier,
        "nonexistent_id_1",
        all_entities[1].identifier if len(all_entities) > 1 else "nonexistent_id_2",
    }
    result = ml_multi_cloud_sample_store.entities_with_identifiers(mixed_ids)

    # Should return only the existing entities
    existing_ids = {
        id for id in mixed_ids if id in {e.identifier for e in all_entities}
    }
    assert len(result) == len(existing_ids)
    assert {e.identifier for e in result} == existing_ids


def test_entities_by_identifiers_with_measurement_results(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test entities_with_identifiers includes measurement results."""
    number_entities = 3
    number_requests = 2
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    # Get entity IDs from the operation
    entity_ids = set()
    for r in requests:
        entity_ids.update({e.identifier for e in r.entities})

    # Fetch entities by identifiers
    retrieved_entities = sample_store.entities_with_identifiers(entity_ids)

    assert len(retrieved_entities) == len(entity_ids)

    # Verify entities have measurement results
    for entity in retrieved_entities:
        assert entity.identifier in entity_ids
        # Entities should have measurement results from the operation
        assert len(entity.measurement_results) > 0


def test_entities_in_operation_empty_operation(
    random_identifier: Callable[[], str],
    ml_multi_cloud_sample_store: SQLSampleStore,
    valid_ado_project_context: ProjectContext,
    ml_multi_cloud_operation_configuration: "DiscoveryOperationResourceConfiguration",
) -> None:
    """Test entities_in_operation with operation that has no entities."""
    from orchestrator.core import OperationResource
    from orchestrator.core.operation.config import DiscoveryOperationEnum
    from orchestrator.metastore.sqlstore import SQLResourceStore

    operation_id = random_identifier()

    # Create an operation with no entities
    sql = SQLResourceStore(project_context=valid_ado_project_context, ensureExists=True)
    sql.addResource(
        OperationResource(
            identifier=operation_id,
            config=ml_multi_cloud_operation_configuration,
            operationType=DiscoveryOperationEnum.SEARCH,
            operatorIdentifier="test-operator",
        )
    )

    # Should return empty list, not raise an error
    result = ml_multi_cloud_sample_store.entities_in_operation(
        operation_id=operation_id
    )
    assert result == []


def test_entities_in_operation_single_operation(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test entities_in_operation returns correct entities for a single operation."""
    number_entities = 5
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    # Get expected entity IDs from requests
    expected_entity_ids = set()
    for r in requests:
        expected_entity_ids.update({e.identifier for e in r.entities})

    # Fetch entities using entities_in_operation
    retrieved_entities = sample_store.entities_in_operation(operation_id=operation_id)

    # Should get all entities from the operation
    retrieved_entity_ids = {e.identifier for e in retrieved_entities}
    assert len(retrieved_entity_ids) == len(expected_entity_ids)
    assert retrieved_entity_ids == expected_entity_ids


def test_entities_in_operation_with_measurement_results(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test entities_in_operation includes measurement results."""
    number_entities = 3
    number_requests = 2
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, _requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    # Fetch entities using entities_in_operation
    retrieved_entities = sample_store.entities_in_operation(operation_id=operation_id)

    # Verify entities have measurement results
    for entity in retrieved_entities:
        assert len(entity.measurement_results) > 0
        # Verify measurement results are valid
        for result in entity.measurement_results:
            assert result.uid is not None
            assert len(result.measurements) > 0


def test_entities_in_operation_deduplication(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test entities_in_operation deduplicates entities when same entity appears in multiple requests."""
    number_entities = 3
    number_requests = 5  # Multiple requests, some entities may repeat
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=number_entities,
            number_requests=number_requests,
            measurements_per_result=measurements_per_result,
            operation_id=operation_id,
        )
    )

    # Get all unique entity IDs from requests
    all_entity_ids = set()
    for r in requests:
        all_entity_ids.update({e.identifier for e in r.entities})

    # Fetch entities using entities_in_operation
    retrieved_entities = sample_store.entities_in_operation(operation_id=operation_id)

    # Should get unique entities (no duplicates)
    retrieved_entity_ids = {e.identifier for e in retrieved_entities}
    assert len(retrieved_entity_ids) == len(retrieved_entities)  # No duplicates
    assert retrieved_entity_ids == all_entity_ids
