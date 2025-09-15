# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, status

from orchestrator.api.routers.v0.actuators import actuators
from orchestrator.api.routers.v0.experiments import experiments

router = APIRouter(
    prefix="/latest",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)

router.include_router(actuators.router)
router.include_router(experiments.router)
