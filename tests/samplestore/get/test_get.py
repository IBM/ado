# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import random
import sqlite3
from collections.abc import Callable

import pandas as pd
import pytest

from orchestrator.core import ADOResource, CoreResourceKinds
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from orchestrator.schema.result import (
    InvalidMeasurementResult,
    ValidMeasurementResult,
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
