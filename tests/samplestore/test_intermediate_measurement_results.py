# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for how SQLSampleStore stores, reads and counts IntermediateMeasurementResults."""

from ado.core.samplestore.base import ExperimentReferencesByMeasurementValidity
from ado.schema.result import (
    IntermediateMeasurementResult,
    InvalidMeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)
from tests.conftest import requires_sqlite_3_38
from tests.samplestore.conftest import (
    OperationWithEachResultType,
    SampleStoreWithEachResultType,
)


@requires_sqlite_3_38
def test_measurement_results_count_for_operation_by_status(
    operation_with_each_result_type: OperationWithEachResultType,
) -> None:
    """Each status filter counts only the results of that type."""

    sample_store = operation_with_each_result_type.sample_store
    operation_id = operation_with_each_result_type.operation_id

    assert (
        sample_store.measurement_results_count_for_operation(operation_id=operation_id)
        == 3
    )
    for status in MeasurementResultStateEnum:
        assert (
            sample_store.measurement_results_count_for_operation(
                operation_id=operation_id, status_filter=status
            )
            == 1
        ), status


def test_operation_entity_statistics_excludes_intermediate_results(
    operation_with_each_result_type: OperationWithEachResultType,
) -> None:
    """An entity whose only result is intermediate has no successful measurement."""

    statistics = (
        operation_with_each_result_type.sample_store.operation_entity_statistics(
            operation_id=operation_with_each_result_type.operation_id
        )
    )

    assert statistics == {
        "entities_with_all_successful_measurements": 1,
        "entities_with_at_least_one_successful_measurement": 1,
        "total_entities": 3,
    }


@requires_sqlite_3_38
def test_operation_measurement_statistics_counts_intermediate_results_separately(
    operation_with_each_result_type: OperationWithEachResultType,
) -> None:
    """Intermediate results are counted apart from successful and failed results."""

    (statistics,) = (
        operation_with_each_result_type.sample_store.operation_measurement_statistics(
            operation_ids={operation_with_each_result_type.operation_id}
        )
    )

    assert statistics.total_results == 3
    assert statistics.successful_results == 1
    assert statistics.failed_results == 1
    assert statistics.intermediate_results == 1
    assert statistics.measured_entities == 3


def test_measurement_results_for_operation_returns_each_result_type(
    operation_with_each_result_type: OperationWithEachResultType,
) -> None:
    """Stored results are read back as their own types."""

    results = (
        operation_with_each_result_type.sample_store.measurement_results_for_operation(
            operation_id=operation_with_each_result_type.operation_id
        )
    )

    assert {result.uid: type(result) for result in results} == {
        operation_with_each_result_type.valid_result.uid: ValidMeasurementResult,
        operation_with_each_result_type.invalid_result.uid: InvalidMeasurementResult,
        operation_with_each_result_type.intermediate_result.uid: IntermediateMeasurementResult,
    }


def test_measurement_requests_for_operation_returns_each_result_type(
    operation_with_each_result_type: OperationWithEachResultType,
) -> None:
    """Results attached to stored requests are read back as their own types."""

    requests = (
        operation_with_each_result_type.sample_store.measurement_requests_for_operation(
            operation_id=operation_with_each_result_type.operation_id
        )
    )

    assert {
        result.uid: type(result)
        for request in requests
        for result in request.measurements
    } == {
        operation_with_each_result_type.valid_result.uid: ValidMeasurementResult,
        operation_with_each_result_type.invalid_result.uid: InvalidMeasurementResult,
        operation_with_each_result_type.intermediate_result.uid: IntermediateMeasurementResult,
    }


def test_entities_with_valid_measurements_excludes_intermediate_results(
    sample_store_with_each_result_type: SampleStoreWithEachResultType,
) -> None:
    """Only the entity with a valid result is returned."""

    stored = sample_store_with_each_result_type

    assert stored.sample_store.entities_with_valid_measurements(
        {
            stored.valid_result.entityIdentifier,
            stored.invalid_result.entityIdentifier,
            stored.intermediate_result.entityIdentifier,
        }
    ) == {stored.valid_result.entityIdentifier}


def test_experiment_references_by_entity_separates_intermediate_results(
    sample_store_with_each_result_type: SampleStoreWithEachResultType,
) -> None:
    """The experiment of an intermediate result is listed only under intermediate measurements."""

    stored = sample_store_with_each_result_type

    references_by_entity = stored.sample_store.experiment_references_by_entity(
        {
            stored.valid_result.entityIdentifier,
            stored.invalid_result.entityIdentifier,
            stored.intermediate_result.entityIdentifier,
        }
    )

    assert references_by_entity == {
        stored.valid_result.entityIdentifier: ExperimentReferencesByMeasurementValidity(
            with_valid_measurements={stored.valid_result.experimentReference}
        ),
        stored.invalid_result.entityIdentifier: ExperimentReferencesByMeasurementValidity(
            with_invalid_measurements={stored.invalid_result.experimentReference}
        ),
        stored.intermediate_result.entityIdentifier: ExperimentReferencesByMeasurementValidity(
            with_intermediate_measurements={
                stored.intermediate_result.experimentReference
            }
        ),
    }


def test_get_entities_omits_intermediate_results(
    sample_store_with_each_result_type: SampleStoreWithEachResultType,
) -> None:
    """Entities loaded from the store hold their valid results but not intermediate ones."""

    stored = sample_store_with_each_result_type

    entities_by_identifier = {
        entity.identifier: entity
        for entity in stored.sample_store.get_entities(
            identifiers={
                stored.valid_result.entityIdentifier,
                stored.intermediate_result.entityIdentifier,
            },
            require_measurements=True,
            refresh=True,
        )
    }

    assert [
        result.uid
        for result in entities_by_identifier[
            stored.valid_result.entityIdentifier
        ].measurement_results
    ] == [stored.valid_result.uid]
    assert (
        entities_by_identifier[
            stored.intermediate_result.entityIdentifier
        ].measurement_results
        == []
    )
