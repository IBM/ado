# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, status

from orchestrator.api.routers.v0.actuators.experiments import experiments
from orchestrator.modules.actuators.registry import ActuatorRegistry

router = APIRouter(
    prefix="/actuators",
    tags=["actuators"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)

router.include_router(experiments.router)


@router.get("")
async def list_actuators() -> list[str]:
    """Retrieve a list of actuator identifiers registered in the system.

    Returns:
        list[str]: A list containing the identifiers of all registered
        actuators.
    """
    return list(ActuatorRegistry().actuatorIdentifierMap.keys())
