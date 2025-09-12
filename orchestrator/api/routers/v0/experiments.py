# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import itertools

from fastapi import APIRouter, HTTPException, status

from orchestrator.api.state.in_memory_requests_storage import (
    get_all_requests_in_memory_storage,
    get_request_in_memory_storage,
)
from orchestrator.cli.exceptions.actuators import (
    NoActuatorWithExperimentError,
    TooManyActuatorsWithExperimentError,
)
from orchestrator.cli.utils.resources.experiments import get_actuator_from_experiment_id
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest

router = APIRouter(
    prefix="/experiments",
    tags=["experiments"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)


@router.get("", tags=["experiments"])
async def get_experiments() -> list[Experiment]:
    actuator_registry = ActuatorRegistry()
    return itertools.chain.from_iterable(
        [
            actuator_registry.catalogForActuatorIdentifier(actuator_id).experiments
            for actuator_id in actuator_registry.actuatorIdentifierMap
        ]
    )


@router.get("/{experiment_identifier}", tags=["experiments"])
async def get_single_experiment(experiment_identifier: str) -> Experiment:
    actuator_registry = ActuatorRegistry()

    try:
        actuator_identifier = get_actuator_from_experiment_id(experiment_identifier)
    except NoActuatorWithExperimentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No actuator implements {experiment_identifier}",
        )
    except TooManyActuatorsWithExperimentError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment {experiment_identifier} is not unique",
        )

    return actuator_registry.experimentForReference(
        ExperimentReference(
            experimentIdentifier=experiment_identifier,
            actuatorIdentifier=actuator_identifier,
        )
    )


@router.get(
    "/{experiment_identifier}/requests",
    tags=["experiments", "requests"],
)
async def get_measurement_requests_for_experiment(
    experiment_identifier: str,
) -> list[MeasurementRequest]:

    return get_all_requests_in_memory_storage(experiment_identifier)


@router.get(
    "/{experiment_identifier}/requests/{request_id}",
    tags=["experiments", "requests"],
)
async def get_single_measurement_request_for_experiment(
    experiment_identifier: str, request_id: str
) -> MeasurementRequest:

    return get_request_in_memory_storage(
        experiment_id=experiment_identifier, request_id=request_id
    )
