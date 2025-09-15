# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from typing import Annotated

from fastapi import Depends, HTTPException, status

from orchestrator.modules.actuators.registry import ActuatorRegistry


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
    actuator = ActuatorRegistry().actuatorForIdentifier(actuatorid=actuator_id)
    available_experiments_for_actuator = {
        experiment.identifier for experiment in actuator.catalog().experiments
    }

    if experiment_id not in available_experiments_for_actuator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Actuator {actuator_id} does not have experiment {experiment_id}",
        )

    return experiment_id
