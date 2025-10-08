# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import ray
from fastapi import APIRouter, Depends, status

from orchestrator.api.dependencies.validation import (
    validated_actuator_id,
    validated_entities_for_experiment,
    validated_experiment_id,
)
from orchestrator.schema.entity import Entity
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest

router = APIRouter(
    prefix="/{experiment_id}/requests",
    dependencies=[Depends(validated_actuator_id), Depends(validated_experiment_id)],
    tags=["actuators"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)


@router.get(
    "",
)
async def list_requests_for_experiment(
    actuator_id: str,
    experiment_id: str,
) -> list[MeasurementRequest]:
    """List all measurement requests for a given experiment.

    Args:
        actuator_id: Identifier of the actuator.
        experiment_id: Identifier of the experiment.

    Returns:
        A list of :class:`MeasurementRequest` objects representing
        the requests stored in memory for the specified experiment.
    """

    return await ray.get_actor(
        name="QueueMonitorActor", namespace="api"
    ).get_measurement_requests.remote(
        experiment_reference=ExperimentReference(
            experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
        )
    )


@router.post("", dependencies=[Depends(validated_entities_for_experiment)])
async def create_experiment_request(
    actuator_id: str, experiment_id: str, entities: list[Entity]
) -> list[str]:
    """Create a new measurement request for an experiment.

    This endpoint submits a list of entities to the actuator actor which
    will process them asynchronously.

    Args:
        actuator_id: Identifier of the actuator.
        experiment_id: Identifier of the experiment.
        entities: A list of :class:`Entity` objects to be measured.

    Returns:
        A list of strings representing the IDs of the submitted requests.
    """
    actuator_actor = await ray.get_actor(
        name="ActuatorDictionaryActor", namespace="api"
    ).get_actuator_actor.remote(actuator_id=actuator_id)

    return await actuator_actor.submit.remote(
        entities=entities,
        experimentReference=ExperimentReference(
            experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
        ),
        requesterid="api",
        requestIndex=0,
    )


@router.get(
    "/{request_id}",
)
async def get_experiment_request_by_id(
    actuator_id: str, experiment_id: str, request_id: str
) -> MeasurementRequest:
    """Retrieve a specific measurement request by its ID.

    Args:
        actuator_id: Identifier of the actuator.
        experiment_id: Identifier of the experiment.
        request_id: Unique identifier of the requested measurement.

    Returns:
        A :class:`MeasurementRequest` instance corresponding to the
        provided request ID. Raises an HTTP 404 if the request is not
        found in the in-memory storage.
    """

    return await ray.get_actor(
        name="QueueMonitorActor", namespace="api"
    ).get_measurement_request_by_id.remote(
        experiment_reference=ExperimentReference(
            experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
        ),
        request_id=request_id,
    )
