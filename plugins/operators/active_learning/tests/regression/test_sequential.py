# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for sequential predictive scheduling with dependent experiments."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from queue import Queue

import pytest
from active_learning.regression._sequential import _SequentialPredictiveWalk
from active_learning.regression.pkh.parameters import PKHParameters
from active_learning.regression.pkh.sampler import PKHSampleSelector
from test_samplers import (
    _adaptive_test_space,
    _entity_with_label,
    _RemoteDiscoverySpace,
)

from ado.modules.operators.randomwalk import RandomWalkParameters
from ado.schema.entity import Entity
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.property import AbstractPropertyDescriptor
from ado.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from ado.schema.result import InvalidMeasurementResult, ValidMeasurementResult


def _walk(requests: int, max_retries: int = 1) -> _SequentialPredictiveWalk:
    """Initialize local scheduler state for the first sampled entity."""

    walk = object.__new__(_SequentialPredictiveWalk)
    parameters = RandomWalkParameters(numberEntities=3, maxRetries=max_retries)
    walk.params = RandomWalkParameters.model_validate(parameters.model_dump())
    walk.log = logging.getLogger("test-sequential-predictive-walk")
    walk.runid = "sequential-test"
    walk._entitiesSampled = 1
    walk._experimentsRequested = requests
    walk._finished_requests = 0
    walk._retriedExperimentRequests = {}
    return walk


def _completion(
    walk: _SequentialPredictiveWalk,
    experiment: Experiment,
    entity: Entity,
    *,
    success: bool = True,
) -> MeasurementRequest:
    """Create a measurement completion with a scalar result or a recorded failure."""

    measurements = (
        (
            ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[
                    ObservedPropertyValue(property=prop, value=2.0)
                    for prop in experiment.observedProperties
                ],
            ),
        )
        if success
        else (
            InvalidMeasurementResult(
                entityIdentifier=entity.identifier,
                experimentReference=experiment.reference,
                reason="Experiment failed during the scheduling regression test",
            ),
        )
    )
    request = MeasurementRequest(
        operation_id=walk.operationIdentifier(),
        requestIndex=0,
        experimentReference=experiment.reference,
        entities=[entity],
        status=(
            MeasurementRequestStateEnum.SUCCESS
            if success
            else MeasurementRequestStateEnum.FAILED
        ),
        measurements=measurements,
    )
    return MeasurementRequest.model_validate(request.model_dump())


async def _sample(
    walk: _SequentialPredictiveWalk,
    queue: Queue,
    iterator: AsyncGenerator[list[Entity], None],
    measurement_space: MeasurementSpace,
) -> list[Entity]:
    """Bound a scheduler step so blocked acquisition becomes a test failure."""

    return await asyncio.wait_for(
        walk._sampleEntityAndAddMeasurementsToQueue(queue, iterator, measurement_space),
        timeout=0.5,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_prerequisite", [False, True])
async def test_unrelated_completion_keeps_dependencies_and_retries_schedulable(
    retry_prerequisite: bool,
) -> None:
    """An unrelated result cannot block the completion that enables the target."""

    space, store, _, _ = _adaptive_test_space()
    unrelated = Experiment(
        actuatorIdentifier="test",
        identifier="unrelated",
        targetProperties=[AbstractPropertyDescriptor(identifier="other")],
    )
    prerequisite = Experiment(
        actuatorIdentifier="test",
        identifier="prerequisite",
        targetProperties=[AbstractPropertyDescriptor(identifier="input")],
    )
    target = Experiment(
        actuatorIdentifier="test",
        identifier="target",
        requiredProperties=(prerequisite.observedProperties[0],),
        targetProperties=[AbstractPropertyDescriptor(identifier="latency")],
    )
    space.measurementSpace = MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(
            experiments=[unrelated, prerequisite, target]
        )
    )
    store.entities.clear()
    selector = PKHSampleSelector(
        PKHParameters(
            targetOutput="latency",
            nJobs=1,
            labelWaitTimeoutSeconds=0.05,
            labelPollIntervalSeconds=0.001,
        )
    )
    iterator = await selector.remoteEntityIterator(_RemoteDiscoverySpace(space))
    first = (await anext(iterator))[0]
    walk = _walk(requests=2)
    queue = Queue()
    try:
        assert walk._processCompletedMeasurement(
            _completion(walk, unrelated, first), space.measurementSpace, queue
        )
        assert queue.empty()
        reads = len(store.reads)
        assert await _sample(walk, queue, iterator, space.measurementSpace) == []
        assert len(store.reads) == reads
        assert walk.numberEntitiesSampled() == 1
        assert walk.numberMeasurementsRequested() == 2

        if retry_prerequisite:
            assert not walk._processCompletedMeasurement(
                _completion(walk, prerequisite, first, success=False),
                space.measurementSpace,
                queue,
            )
            retry = queue.get_nowait()
            assert retry["experimentReference"] == prerequisite.reference
            walk._experimentsRequested += 1
            assert await _sample(walk, queue, iterator, space.measurementSpace) == []

        assert walk._processCompletedMeasurement(
            _completion(walk, prerequisite, first), space.measurementSpace, queue
        )
        dependent = queue.get_nowait()
        assert dependent["experimentReference"] == target.reference
        assert dependent["requestIndex"] == 0
        assert queue.empty()
        walk._experimentsRequested += 1
        assert await _sample(walk, queue, iterator, space.measurementSpace) == []

        finished = _completion(walk, target, dependent["entities"][0])
        store.entities[first.identifier] = finished.entities[0]
        assert walk._processCompletedMeasurement(
            finished, space.measurementSpace, queue
        )
        second = await _sample(walk, queue, iterator, space.measurementSpace)
        assert len(second) == 1
        assert second[0].identifier != first.identifier
        assert walk.numberEntitiesSampled() == 2
        assert walk.numberMeasurementsRequested() == 3 + int(retry_prerequisite)
        assert queue.qsize() == 2
        requests = [queue.get_nowait() for _ in range(2)]
        assert {request["requestIndex"] for request in requests} == {1}
        assert {request["experimentReference"] for request in requests} == {
            unrelated.reference,
            prerequisite.reference,
        }
    finally:
        await iterator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("exhaust_pool", [False, True])
async def test_single_target_feedback_resumes_or_exhausts_pool(
    exhaust_pool: bool,
) -> None:
    """A target completion permits the next query or clean finite-pool exhaustion."""

    space, store, target, entities = _adaptive_test_space()
    if exhaust_pool:
        store.entities = {
            entity.identifier: _entity_with_label(entity, target, float(index))
            for index, entity in enumerate(entities[:-1])
        }
    selector = PKHSampleSelector(
        PKHParameters(
            targetOutput="latency",
            nJobs=1,
            labelWaitTimeoutSeconds=0.05,
            labelPollIntervalSeconds=0.001,
        )
    )
    iterator = await selector.remoteEntityIterator(_RemoteDiscoverySpace(space))
    first = (await anext(iterator))[0]
    walk = _walk(requests=1)
    queue = Queue()
    try:
        assert await _sample(walk, queue, iterator, space.measurementSpace) == []
        assert walk.numberEntitiesSampled() == 1
        assert walk.numberMeasurementsRequested() == 1
        finished = _completion(walk, space.measurementSpace.experiments[0], first)
        store.entities[first.identifier] = finished.entities[0]
        assert walk._processCompletedMeasurement(
            finished, space.measurementSpace, queue
        )
        selected = await _sample(walk, queue, iterator, space.measurementSpace)
        assert len(selected) == int(not exhaust_pool)
        assert walk.numberEntitiesSampled() == 1 + int(not exhaust_pool)
        assert walk.numberMeasurementsRequested() == 1
        assert queue.qsize() == int(not exhaust_pool)
    finally:
        await iterator.aclose()


def test_unrelated_operation_completion_does_not_release_feedback() -> None:
    """Only this operation's completions count toward its outstanding requests."""

    space, _, _, entities = _adaptive_test_space()
    walk = _walk(requests=1)
    request = _completion(walk, space.measurementSpace.experiments[0], entities[0])
    request = request.model_copy(update={"operation_id": "another-operation"})
    walk._processCompletedMeasurement(request, space.measurementSpace, Queue())
    assert walk._finished_requests == 0


@pytest.mark.parametrize("max_retries", [0, 1])
def test_terminal_measurement_failure_raises_after_retry_limit(
    max_retries: int,
) -> None:
    """An exhausted failure surfaces even for the final budgeted entity."""

    space, _, _, entities = _adaptive_test_space()
    walk = _walk(requests=1, max_retries=max_retries)
    walk.params.numberEntities = 1
    queue = Queue()
    experiment = space.measurementSpace.experiments[0]
    if max_retries:
        assert not walk._processCompletedMeasurement(
            _completion(walk, experiment, entities[0], success=False),
            space.measurementSpace,
            queue,
        )
        assert queue.get_nowait()["experimentReference"] == experiment.reference
        walk._experimentsRequested += 1
    with pytest.raises(RuntimeError, match="retry") as caught:
        walk._processCompletedMeasurement(
            _completion(walk, experiment, entities[0], success=False),
            space.measurementSpace,
            queue,
        )
    assert experiment.identifier in str(caught.value)
    assert walk._finished_requests == 1 + max_retries
    assert queue.empty()
