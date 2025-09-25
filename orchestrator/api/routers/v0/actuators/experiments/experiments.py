# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, Depends, status

from orchestrator.api.dependencies.validation import (
    validated_actuator_id,
    validated_experiment_id,
)
from orchestrator.api.routers.v0.actuators.experiments.requests import requests
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.reference import ExperimentReference

router = APIRouter(
    prefix="/{actuator_id}/experiments",
    dependencies=[Depends(validated_actuator_id)],
    tags=["actuators"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)

router.include_router(requests.router)


@router.get("")
async def list_actuator_experiments(actuator_id: str) -> list[Experiment]:
    """List all experiments for a given actuator.

    Args:
        actuator_id (str): The unique identifier of the actuator.

    Returns:
        list[Experiment]: A list of experiments associated with the actuator.
    """
    actuator_registry = ActuatorRegistry()
    return (
        actuator_registry.actuatorForIdentifier(actuatorid=actuator_id)
        .catalog()
        .experiments
    )


@router.get("/{experiment_id}", dependencies=[Depends(validated_experiment_id)])
async def get_actuator_experiment_by_id(
    actuator_id: str, experiment_id: str
) -> Experiment:
    """Retrieve a specific experiment by its identifier for a given actuator.

    Args:
        actuator_id (str): The unique identifier of the actuator.
        experiment_id (str): The unique identifier of the experiment.

    Returns:
        Experiment: The corresponding Experiment instance.
    """
    return ActuatorRegistry().experimentForReference(
        ExperimentReference(
            experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
        )
    )
