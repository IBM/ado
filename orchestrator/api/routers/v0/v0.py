"""
Module providing FastAPI routers for v0 of the Orchestrator API.

This module defines the base :class:`fastapi.APIRouter` instance for the
``/v0`` API namespace and includes the individual routers for the
``actuators`` and ``experiments`` submodules.
"""

# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, status

from orchestrator.api.routers.v0.actuators import actuators
from orchestrator.api.routers.v0.experiments import experiments

# Create the parent router for the v0 API version.
router: APIRouter = APIRouter(
    prefix="/v0",
    tags=["v0"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)

# Register sub-routers.
router.include_router(actuators.router)
router.include_router(experiments.router)
