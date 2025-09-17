# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, Depends, status

from orchestrator.api.dependencies.validation import (
    validated_actuator_id,
    validated_experiment_id,
)
from orchestrator.api.state.actuator_actors import (
    get_actuator_actor,
)
from orchestrator.api.state.in_memory_requests_storage import (
    get_all_requests_in_memory_storage,
    get_request_in_memory_storage,
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

    return get_all_requests_in_memory_storage(
        experiment_reference=ExperimentReference(
            experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
        )
    )


@router.post(
    "",
)
async def create_experiment_request(
    actuator_id: str, experiment_id: str, entities: list[Entity]
) -> list[str]:

    return await get_actuator_actor(actuator_id).submit.remote(
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

    return get_request_in_memory_storage(
        experiment_reference=ExperimentReference(
            experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
        ),
        request_id=request_id,
    )
