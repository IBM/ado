# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import uuid
from collections.abc import Callable

import pydantic
import pytest

from orchestrator.schema.entity import Entity
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property import AbstractPropertyDescriptor
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference
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


# missing test: multiple results per entity, increases results index
def test_invalid_series_representation_format_raises_error(
    valid_measurement_result: ValidMeasurementResult, entity: Entity
) -> None:
    request = MeasurementRequest(
        entities=[entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
        measurements=(valid_measurement_result,),
    )
    with pytest.raises(
        ValueError, match="The only supported series representation output formats are"
    ):
        request.series_representation(output_format="blah")


def test_can_assign_none_cannot_reset(
    valid_measurement_result: ValidMeasurementResult, entity: Entity
) -> None:

    # We do not set a value for the measurement field - this will work
    request = MeasurementRequest(
        entities=[entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )

    # Test one can set the measurements to None
    request.measurements = None

    # Test the measurement cannot be reset
    # NOTE: We may want to allow reassignment after None assignment.
    # I can't imagine a situation where the result should be set to None forever - it should be InvalidMeasurementResult
    with pytest.raises(pydantic.ValidationError):
        request.measurements = [valid_measurement_result]


def test_cannot_assign_empty(
    valid_measurement_result: ValidMeasurementResult, entity: Entity
) -> None:

    # We do not set a value for the measurement field - this will work
    request = MeasurementRequest(
        entities=[entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )

    with pytest.raises(pydantic.ValidationError):
        request.measurements = []


def test_string_representation(
    valid_measurement_result: ValidMeasurementResult,
    invalid_measurement_result: InvalidMeasurementResult,
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:

    import copy

    # Single entity
    request = MeasurementRequest(
        entities=[entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
        measurements=(valid_measurement_result,),
    )

    assert str(
        request
    ) == "request-{}-experiment-{}-entities-{}-requester-{}-time-{}".format(  # noqa: UP032
        request.requestid,
        request.experimentReference.experimentIdentifier,
        entity,
        request.operation_id,
        request.timestamp,
    )

    other_entity = copy.deepcopy(entity)
    other_entity.identifier = "blablabla"

    measurement_result_for_other_entity = ValidMeasurementResult(
        entityIdentifier=other_entity.identifier, measurements=property_values
    )

    request = MeasurementRequest(
        entities=[entity, other_entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )

    # We set a value for the measurement field - this will work
    request.measurements = [
        valid_measurement_result,
        measurement_result_for_other_entity,
    ]

    assert str(
        request
    ) == "request-{}-experiment-{}-entities-multi-{}-requester-{}-time-{}".format(  # noqa: UP032
        request.requestid,
        request.experimentReference.experimentIdentifier,
        2,
        request.operation_id,
        request.timestamp,
    )


def test_string_representation_replayed(
    valid_measurement_result: ValidMeasurementResult,
    invalid_measurement_result: InvalidMeasurementResult,
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:

    import copy

    # Single entity
    request = ReplayedMeasurement(
        entities=[entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
        measurements=(valid_measurement_result,),
    )

    assert str(request) == "{}-experiment-{}-entities-{}-time-{}".format(  # noqa: UP032
        request.requestid,
        request.experimentReference.experimentIdentifier,
        entity,
        request.timestamp,
    )

    other_entity = copy.deepcopy(entity)
    other_entity.identifier = "blablabla"

    measurement_result_for_other_entity = ValidMeasurementResult(
        entityIdentifier=other_entity.identifier, measurements=property_values
    )

    request = ReplayedMeasurement(
        entities=[entity, other_entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )

    # We set a value for the measurement field - this will work
    request.measurements = [
        valid_measurement_result,
        measurement_result_for_other_entity,
    ]

    assert str(
        request
    ) == "{}-experiment-{}-entities-multi-{}-time-{}".format(  # noqa: UP032
        request.requestid,
        request.experimentReference.experimentIdentifier,
        2,
        request.timestamp,
    )


def test_cannot_reassign_measurements_field_in_measurement_request(
    valid_measurement_result: ValidMeasurementResult,
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:
    """This tests that once the measurement field of MeasurementRequest is assigned a set of MeasurementResults
    that set cannot be changed or set to None"""

    import copy

    other_entity = copy.deepcopy(entity)
    other_entity.identifier = "blablabla"
    measurement_result_for_other_entity = ValidMeasurementResult(
        entityIdentifier=other_entity.identifier, measurements=property_values
    )

    # We do not set a value for the measurement field - this will work
    request = MeasurementRequest(
        entities=[entity, other_entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )

    # We attempt to set a result for only one of the two entities in the measurement field - this will fail
    # You need to set values for all entities
    with pytest.raises(pydantic.ValidationError):
        request.measurements = [valid_measurement_result]

    # Set values for both entities
    request.measurements = [
        valid_measurement_result,
        measurement_result_for_other_entity,
    ]

    # We attempt to set a copy of the results - this should work
    request.measurements = [
        valid_measurement_result.model_copy(deep=True),
        measurement_result_for_other_entity.model_copy(deep=True),
    ]

    # We attempt to set a different set of results for the measurement field - this will fail
    # Although the values are the same the result id is different so it should not be allowed
    # Note: Although it raises an error this sets the measurement field to the new value
    different_result_for_other_entity = ValidMeasurementResult(
        entityIdentifier=other_entity.identifier, measurements=property_values
    )

    with pytest.raises(pydantic.ValidationError):
        request.measurements = [
            valid_measurement_result,
            different_result_for_other_entity,
        ]

    # We attempt to set None value for measurement field - this will also fail
    with pytest.raises(pydantic.ValidationError):
        request.measurements = None


def test_measurement_request_measurement_for_entity(
    valid_measurement_result: ValidMeasurementResult, entity: Entity
) -> None:
    request = MeasurementRequest(
        entities=[entity],
        measurements=(valid_measurement_result,),
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )
    assert request.measurement_for_entity(entity.identifier) == valid_measurement_result

    with pytest.raises(
        ValueError,
        match="Entity with identifier incorrect_id was not part of this MeasurementRequest",
    ):
        request.measurement_for_entity("incorrect_id")


def test_measurement_request_valid(
    valid_measurement_result: ValidMeasurementResult, entity: Entity
) -> None:

    MeasurementRequest(
        entities=[entity],
        measurements=(valid_measurement_result,),
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )


def test_measurement_request_invalid(
    invalid_measurement_result: InvalidMeasurementResult, entity: Entity
) -> None:

    MeasurementRequest(
        entities=[entity],
        measurements=(invalid_measurement_result,),
        experimentReference=invalid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )


def test_measurement_request_mismatched_entities(
    valid_measurement_result: ValidMeasurementResult, entity: Entity
) -> None:
    """Tests all entity ids in the request measurements field must have a matching entity in request entities field"""

    # Create a InvalidResult for a different entity
    invalid_result = InvalidMeasurementResult(
        entityIdentifier="test-entity-aaaccc",
        reason="Insufficient memory",
        experimentReference=valid_measurement_result.experimentReference,
    )

    # Should fail because both invalid result references an entity that isn't in entities list
    with pytest.raises(pydantic.ValidationError):

        MeasurementRequest(
            entities=[entity],
            measurements=(
                valid_measurement_result,
                invalid_result,
            ),
            experimentReference=valid_measurement_result.experimentReference,
            requestid="testid-aaaccc",
            requestIndex=0,
            operation_id="pytest",
        )

    # Try also via assignment
    request = MeasurementRequest(
        entities=[entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )

    with pytest.raises(pydantic.ValidationError):

        request.measurements = [valid_measurement_result, invalid_result]


def test_measurement_request_mismatched_experiments(
    entity: Entity, valid_measurement_result: ValidMeasurementResult
) -> None:
    """Tests all experiments in request.measurements match request.experimentReference"""

    # Create a InvalidResult for the same entity with a different experiment
    invalid_result = InvalidMeasurementResult(
        entityIdentifier=entity.identifier,
        reason="Insufficient memory",
        experimentReference=ExperimentReference(
            experimentIdentifier="testexp", actuatorIdentifier="testact"
        ),
    )

    # Should fail because the InvalidResult experiment does not match the request experiment
    with pytest.raises(pydantic.ValidationError):

        MeasurementRequest(
            entities=[entity],
            measurements=(
                valid_measurement_result,
                invalid_result,
            ),
            experimentReference=valid_measurement_result.experimentReference,
            requestid="testid-aaaccc",
            requestIndex=0,
            operation_id="pytest",
        )

    # Try also via assignment
    request = MeasurementRequest(
        entities=[entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )

    with pytest.raises(pydantic.ValidationError):
        request.measurements = [valid_measurement_result, invalid_result]


def test_measurement_request_multiple_entity_measurement(
    valid_measurement_result: ValidMeasurementResult, entity: Entity
) -> None:

    # Tests that the same entity can't have multiple measurement results

    # Create a InvalidResult for the same entity
    invalid_result = InvalidMeasurementResult(
        entityIdentifier=entity.identifier,
        reason="Insufficient memory",
        experimentReference=valid_measurement_result.experimentReference,
    )

    # Should fail because both results reference the same entity
    with pytest.raises(pydantic.ValidationError):
        MeasurementRequest(
            entities=[entity],
            measurements=(valid_measurement_result, invalid_result),
            experimentReference=valid_measurement_result.experimentReference,
            requestid="testid-aaaccc",
            requestIndex=0,
            operation_id="pytest",
        )

    # Try also via assignment
    request = MeasurementRequest(
        entities=[entity],
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )

    with pytest.raises(pydantic.ValidationError):
        request.measurements = [valid_measurement_result, invalid_result]


def test_measurement_request_series_representation(
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
    random_identifier: Callable[[], str],
) -> None:

    number_entities = 2
    measurements_per_result = 1
    operation_id = random_identifier()

    random_request: MeasurementRequest = (
        random_ml_multi_cloud_benchmark_performance_measurement_requests(
            number_entities=number_entities,
            measurements_per_result=measurements_per_result,
            status=MeasurementRequestStateEnum.SUCCESS,
            operation_id=operation_id,
        )
    )

    # Observed only changes the result representation
    target_series_representation = random_request.series_representation(
        output_format="target"
    )

    assert target_series_representation is not None
    assert len(target_series_representation) == number_entities
    occurrences_counter = {}
    for i in range(number_entities):
        series = target_series_representation[i]

        #
        if random_request.entities[i].identifier not in occurrences_counter:
            expected_result_id = 0
        else:
            expected_result_id = occurrences_counter[
                random_request.entities[i].identifier
            ]

        occurrences_counter[random_request.entities[i].identifier] = (
            expected_result_id + 1
        )

        #
        assert series["request_id"] == random_request.requestid
        assert series["request_index"] == random_request.requestIndex
        assert series["result_index"] == expected_result_id
        assert series["entity_index"] == i
        assert series["identifier"] == random_request.entities[i].identifier
        assert series["experiment_id"] == random_request.experimentReference
        assert series["valid"]


def test_populate_measurement_results_in_entities(
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> None:

    random_entity = random_ml_multi_cloud_benchmark_performance_entities(quantity=1)[0]
    shared_uid = uuid.uuid4()

    # We create a measurement result with a set UID and add it to the entity
    first_measurement_result = ValidMeasurementResult(
        uid=str(shared_uid),
        entityIdentifier=random_entity.identifier,
        measurements=[
            ObservedPropertyValue(
                value=1,
                property=ObservedProperty(
                    targetProperty=AbstractPropertyDescriptor(
                        identifier="wallClockRuntime"
                    ),
                    experimentReference=ExperimentReference(
                        experimentIdentifier="benchmark_performance",
                        actuatorIdentifier="replay",
                    ),
                ),
            )
        ],
    )

    random_entity.add_measurement_result(first_measurement_result)

    # We create a second measurement result that has the same UID
    # but a different value
    second_measurement_result = ValidMeasurementResult(
        uid=str(shared_uid),
        entityIdentifier=random_entity.identifier,
        measurements=[
            ObservedPropertyValue(
                value=2,
                property=ObservedProperty(
                    targetProperty=AbstractPropertyDescriptor(
                        identifier="wallClockRuntime"
                    ),
                    experimentReference=ExperimentReference(
                        experimentIdentifier="benchmark_performance",
                        actuatorIdentifier="replay",
                    ),
                ),
            )
        ],
    )
    assert first_measurement_result.uid == second_measurement_result.uid

    # We create a MeasurementRequest
    measurement_request = MeasurementRequest(
        operation_id="test_populate_measurement_results_in_entities",
        requestIndex=0,
        experimentReference=ExperimentReference(
            experimentIdentifier="benchmark_performance",
            actuatorIdentifier="replay",
        ),
        entities=[random_entity],
        requestid="test_populate_measurement_results_in_entities",
    )

    # We set the measurements of the request as the second result
    # This will attempt to populate this result in the entity, but
    # there will be another one with the same UID.
    # The "not in" check should prevent this from being added
    measurement_request.measurements = [second_measurement_result]
    assert not any(
        result.uid == shared_uid
        and result.measurements[0].value
        == second_measurement_result.measurements[0].value
        for result in random_entity.measurement_results
    )


@pytest.fixture
def measurement_request_valid(
    valid_measurement_result: ValidMeasurementResult, entity: Entity
) -> MeasurementRequest:

    return MeasurementRequest(
        entities=[entity],
        measurements=(valid_measurement_result,),
        experimentReference=valid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )


@pytest.fixture
def measurement_request_invalid(
    invalid_measurement_result: InvalidMeasurementResult, entity: Entity
) -> MeasurementRequest:

    return MeasurementRequest(
        entities=[entity],
        measurements=(invalid_measurement_result,),
        experimentReference=invalid_measurement_result.experimentReference,
        requestid="testid-aaaccc",
        requestIndex=0,
        operation_id="pytest",
    )
