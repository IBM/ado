# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, Depends, HTTPException, status

from orchestrator.api.dependencies.validation import validated_actuator_id
from orchestrator.api.routers.v0.actuators.experiments.requests import requests
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import Experiment

router = APIRouter(
    prefix="/{actuator_id}/experiments",
    dependencies=[Depends(validated_actuator_id)],
    tags=["experiments"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)

router.include_router(requests.router)


@router.get("")
async def list_actuator_experiments(actuator_id: str) -> list[Experiment]:
    actuator_registry = ActuatorRegistry()
    return (
        actuator_registry.actuatorForIdentifier(actuatorid=actuator_id)
        .catalog()
        .experiments
    )


@router.get(
    "/{experiment_id}",
)
async def get_actuator_experiment_by_id(
    actuator_id: str, experiment_id: str
) -> Experiment:
    actuator = ActuatorRegistry().actuatorForIdentifier(actuatorid=actuator_id)
    identifier_experiment_map = {
        experiment.identifier: experiment
        for experiment in actuator.catalog().experiments
    }

    if experiment_id not in identifier_experiment_map:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Actuator {actuator_id} does not have experiment {experiment_id}",
        )

    return identifier_experiment_map[experiment_id]
