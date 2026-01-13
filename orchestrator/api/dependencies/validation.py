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
    """
    Validate that the supplied actuator ID exists in the registry.

    Args:
        actuator_id: The identifier of the actuator to validate.

    Returns:
        The validated actuator ID (identical to the input if validation passes).

    Raises:
        HTTPException: If the actuator ID is unknown (404 Not Found).
    """
    if actuator_id not in ActuatorRegistry().actuatorIdentifierMap:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown actuator {actuator_id}",
        )

    return actuator_id


def validated_experiment_id(
    actuator_id: Annotated[str, Depends(validated_actuator_id)], experiment_id: str
) -> str:
    """
    Validate that a given experiment belongs to the specified actuator.

    Args:
        actuator_id: Actuator identifier, already validated by ``validated_actuator_id``.
        experiment_id: Identifier of the experiment to validate.

    Returns:
        The validated experiment ID (identical to the input if validation passes).

    Raises:
        HTTPException: If the experiment is not associated with the actuator
            (404 Not Found).
    """
    try:
        ActuatorRegistry().experimentForReference(
            ExperimentReference(
                experimentIdentifier=experiment_id, actuatorIdentifier=actuator_id
            )
        )
    except UnknownExperimentError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Actuator {actuator_id} does not have experiment {experiment_id}",
        ) from e

    return experiment_id


def validated_entities_for_experiment(
    actuator_id: Annotated[str, Depends(validated_actuator_id)],
    experiment_id: Annotated[str, Depends(validated_experiment_id)],
    entities: list[Entity],
) -> list[Entity]:
    """
    Validate that the provided entities are valid for a specific experiment.

    Args:
        actuator_id: Actuator identifier, already validated by ``validated_actuator_id``.
        experiment_id: Experiment identifier, already validated by ``validated_experiment_id``.
        entities: A list of entities to validate against the experiment.

    Returns:
        The list of validated entities (identical to the input if validation passes).

    Raises:
        HTTPException: If any entity is invalid for the experiment (422 Unprocessable Entity).
    """
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
        ) from e

    return entities
