# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from typing import Annotated

from fastapi import Depends, HTTPException, status

from orchestrator.modules.actuators.registry import (
    ActuatorRegistry,
    UnknownExperimentError,
)
from orchestrator.schema.entity import Entity
from orchestrator.schema.reference import ExperimentReference


def validated_actuator_id(actuator_id: str) -> str:
    if actuator_id not in ActuatorRegistry().actuatorIdentifierMap:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown actuator {actuator_id}",
        )

    return actuator_id


def validated_experiment_id(
    actuator_id: Annotated[str, Depends(validated_actuator_id)], experiment_id: str
) -> str:
    try:
        ActuatorRegistry().experimentForReference(
            ExperimentReference(
                experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
            )
        )
    except UnknownExperimentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Actuator {actuator_id} does not have experiment {experiment_id}",
        )

    return experiment_id


def entity_is_valid_for_experiment(
    actuator_id: str, experiment_id: str, entities: list[Entity]
):
    requested_experiment = ActuatorRegistry().experimentForReference(
        ExperimentReference(
            experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
        )
    )

    try:
        for entity in entities:
            requested_experiment.propertyValuesFromEntity(entity)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
