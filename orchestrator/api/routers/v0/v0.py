# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, status

from orchestrator.api.routers.v0.actuators import actuators

router = APIRouter(
    prefix="/v0",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)

router.include_router(actuators.router)
