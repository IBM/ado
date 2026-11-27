# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import pytest

from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.entity import Entity
from ado.schema.request import (
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from ado.schema.result import (
    IntermediateMeasurementResult,
    InvalidMeasurementResult,
    MeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)

if TYPE_CHECKING:
    from ado.core.operation.config import DiscoveryOperationResourceConfiguration


class OperationWithEachResultType(NamedTuple):
    """An operation stored in a sample store with one result of each type.

    Each result belongs to its own request and its own entity.
    """

    sample_store: SQLSampleStore
    operation_id: str
    valid_result: ValidMeasurementResult
    invalid_result: InvalidMeasurementResult
    intermediate_result: IntermediateMeasurementResult


@pytest.fixture
def operation_with_each_result_type(
    ml_multi_cloud_sample_store: SQLSampleStore,
    valid_ado_project_context: ProjectContext,
    ml_multi_cloud_operation_configuration: "DiscoveryOperationResourceConfiguration",
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
    random_identifier: Callable[[], str],
) -> OperationWithEachResultType:
    """Store an operation with one valid, one invalid and one intermediate result.

    Each result is held by its own request and measures its own entity.

    Args:
        ml_multi_cloud_sample_store: The sample store the requests are added to.
        valid_ado_project_context: The project context holding the operation.
        ml_multi_cloud_operation_configuration: The configuration of the operation.
        random_ml_multi_cloud_benchmark_performance_entities: Fixture creating
            distinct entities.
        random_ml_multi_cloud_benchmark_performance_measurement_results: Fixture
            creating results with a given status.
        random_identifier: Fixture creating random identifiers.

    Returns:
        The sample store, the operation identifier and the three stored results.
    """
    from ado.core import OperationResource
    from ado.core.operation.config import DiscoveryOperationEnum
    from ado.metastore.sqlstore import SQLResourceStore

    operation_id = random_identifier()
    SQLResourceStore(
        project_context=valid_ado_project_context
    ).addResourceWithRelationships(
        OperationResource(
            identifier=operation_id,
            config=ml_multi_cloud_operation_configuration,
            operationType=DiscoveryOperationEnum.EXPLORE,
            operatorIdentifier="test-operator",
        ),
        relatedIdentifiers=ml_multi_cloud_operation_configuration.spaces,
    )

    request_status_for_result_status = {
        MeasurementResultStateEnum.VALID: MeasurementRequestStateEnum.SUCCESS,
        MeasurementResultStateEnum.INVALID: MeasurementRequestStateEnum.FAILED,
        MeasurementResultStateEnum.INTERMEDIATE: MeasurementRequestStateEnum.UNKNOWN,
    }
    entities = random_ml_multi_cloud_benchmark_performance_entities(
        len(request_status_for_result_status)
    )

    results_by_status = {}
    for request_index, (entity, (result_status, request_status)) in enumerate(
        zip(entities, request_status_for_result_status.items(), strict=True)
    ):
        result = random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, result_status
        )
        request = ReplayedMeasurement(
            operation_id=operation_id,
            requestIndex=request_index,
            experimentReference=result.experimentReference,
            entities=[entity],
            requestid=random_identifier(),
            status=request_status,
            measurements=(result,),
        )
        request_db_id = ml_multi_cloud_sample_store.add_measurement_request(
            request=request
        )
        ml_multi_cloud_sample_store.add_measurement_results(
            results=[result],
            skip_relationship_to_request=False,
            request_db_id=request_db_id,
        )
        results_by_status[result_status] = result

    return OperationWithEachResultType(
        sample_store=ml_multi_cloud_sample_store,
        operation_id=operation_id,
        valid_result=results_by_status[MeasurementResultStateEnum.VALID],
        invalid_result=results_by_status[MeasurementResultStateEnum.INVALID],
        intermediate_result=results_by_status[MeasurementResultStateEnum.INTERMEDIATE],
    )


class SampleStoreWithEachResultType(NamedTuple):
    """A sample store holding only one result of each type, each for its own entity."""

    sample_store: SQLSampleStore
    valid_result: ValidMeasurementResult
    invalid_result: InvalidMeasurementResult
    intermediate_result: IntermediateMeasurementResult


@pytest.fixture
def sample_store_with_each_result_type(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> SampleStoreWithEachResultType:
    """Store one valid, one invalid and one intermediate result in an empty sample store.

    Args:
        random_sql_sample_store: Fixture creating empty SQL sample stores.
        random_ml_multi_cloud_benchmark_performance_entities: Fixture creating
            entities.
        random_ml_multi_cloud_benchmark_performance_measurement_results: Fixture
            creating results with a given status.

    Returns:
        The sample store and the three stored results.
    """

    sample_store = random_sql_sample_store()
    entities = random_ml_multi_cloud_benchmark_performance_entities(3)
    sample_store.addEntities(entities)

    valid_result, invalid_result, intermediate_result = (
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            entity, 1, status
        )
        for entity, status in zip(
            entities,
            [
                MeasurementResultStateEnum.VALID,
                MeasurementResultStateEnum.INVALID,
                MeasurementResultStateEnum.INTERMEDIATE,
            ],
            strict=True,
        )
    )
    sample_store.add_measurement_results(
        [valid_result, invalid_result, intermediate_result],
        skip_relationship_to_request=True,
    )

    return SampleStoreWithEachResultType(
        sample_store=sample_store,
        valid_result=valid_result,
        invalid_result=invalid_result,
        intermediate_result=intermediate_result,
    )
