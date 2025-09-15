# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, Depends, HTTPException, status

from orchestrator.api.dependencies.validation import (
    validated_actuator_id,
    validated_experiment_id,
)
from orchestrator.api.state.actuator_actors import (
    get_actuator_actor,
    set_actuator_actor,
)
from orchestrator.api.state.in_memory_requests_storage import requests_memory_storage
from orchestrator.api.state.queue import shared_queue
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.entity import Entity
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest

router = APIRouter(
    prefix="/requests",
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

    experiment_reference = ExperimentReference(
        experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
    )
    if experiment_reference not in requests_memory_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No requests associated with {experiment_reference}",
        )

    return list(requests_memory_storage[experiment_reference].values())


@router.post(
    "",
)
async def create_experiment_request(
    actuator_id: str, experiment_id: str, entities: list[Entity]
) -> list[str]:

    experiment_reference = ExperimentReference(
        experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
    )
    actuator = get_actuator_actor(actuator_id)
    if actuator is None:
        actuator = (
            ActuatorRegistry()
            .actuatorForIdentifier(actuatorid=actuator_id)
            .options(name=actuator_id, namespace="api")
            .remote(queue=shared_queue, params=None)
        )
        set_actuator_actor(actuator_identifier=actuator_id, actuator_actor=actuator)

    return await actuator.submit.remote(
        entities=entities,
        experimentReference=experiment_reference,
        requesterid="api",
        requestIndex=0,
    )


@router.get(
    "/{request_id}",
)
async def get_experiment_request_by_id(
    actuator_id: str, experiment_id: str, request_id: str
) -> MeasurementRequest:
    experiment_reference = ExperimentReference(
        experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
    )
    if (
        experiment_reference not in requests_memory_storage
        or request_id not in requests_memory_storage[experiment_reference]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Requests {request_id} not found for {experiment_reference}",
        )

    return requests_memory_storage[experiment_reference][request_id]
