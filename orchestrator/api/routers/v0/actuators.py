from fastapi import APIRouter, HTTPException, status

from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import Experiment

router = APIRouter(
    prefix="/actuators",
    tags=["actuators"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)


@router.get("", tags=["actuators"])
async def get_actuators() -> list[str]:
    return list(ActuatorRegistry.globalRegistry().actuatorIdentifierMap.keys())


@router.get("/{actuator_identifier}/experiments", tags=["actuators", "experiments"])
async def get_actuator_experiments(actuator_identifier: str) -> list[Experiment]:
    actuator_registry = ActuatorRegistry.globalRegistry()
    if actuator_identifier not in actuator_registry.actuatorIdentifierMap:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown actuator {actuator_identifier}",
        )

    return (
        actuator_registry.actuatorForIdentifier(actuatorid=actuator_identifier)
        .catalog()
        .experiments
    )


@router.get(
    "/{actuator_identifier}/experiments/{experiment_identifier}",
    tags=["actuators", "experiments"],
)
async def get_actuator_single_experiment(
    actuator_identifier: str, experiment_identifier: str
) -> Experiment:
    actuator_registry = ActuatorRegistry.globalRegistry()
    if actuator_identifier not in actuator_registry.actuatorIdentifierMap:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown actuator {actuator_identifier}",
        )

    actuator = actuator_registry.actuatorForIdentifier(actuatorid=actuator_identifier)
    identifier_experiment_map = {
        experiment.identifier: experiment
        for experiment in actuator.catalog().experiments
    }

    if experiment_identifier not in identifier_experiment_map:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Actuator {actuator_identifier} does not have experiment {experiment_identifier}",
        )

    return identifier_experiment_map[experiment_identifier]
