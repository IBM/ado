"""
Module providing FastAPI routers for the latest version of the Orchestrator API.

This module defines the base :class:`fastapi.APIRouter` instance for the
``latest`` API namespace and includes the individual routers for the
``actuators`` and ``experiments`` submodules.
"""

# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import APIRouter, status

from orchestrator.api.routers.v0.actuators import actuators
from orchestrator.api.routers.v0.experiments import experiments

# Create the parent router for the latest API version.
router = APIRouter(
    prefix="/latest",
    tags=["latest"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)

# Register sub-routers.
router.include_router(actuators.router)
router.include_router(experiments.router)
