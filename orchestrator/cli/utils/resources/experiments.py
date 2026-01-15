# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.exceptions.actuators import (
    ActuatorDoesNotHaveExperimentError,
    NoActuatorWithExperimentError,
    TooManyActuatorsWithExperimentError,
)
from orchestrator.cli.utils.output.prints import ERROR, HINT, console_print, magenta
from orchestrator.modules.actuators.registry import ActuatorRegistry


def get_actuators_implementing_experiment(experiment_id: str) -> set[str]:
    """
    Returns a set of actuators that implement a given experiment.

    Args:
        experiment_id (str): The identifier of the experiment.

    Returns:
        set[str]: A set of actuator identifiers that implement the experiment.
    """
    registry = ActuatorRegistry.globalRegistry()
    actuators_with_target_experiment: set[str] = set()

    for actuator_id in registry.actuatorIdentifierMap:
        catalog = registry.catalogForActuatorIdentifier(actuator_id)
        for experiment in catalog.experiments:
            if experiment.identifier == experiment_id:
                actuators_with_target_experiment.add(actuator_id)

    return actuators_with_target_experiment


def get_actuators_from_experiment_id(
    experiment_id: str,
) -> set[str]:
    """
    Retrieves a set of actuators that implement a given experiment ID.

    Args:
        experiment_id (str): The ID of the experiment.

    Returns:
        set[str]: A set of actuator identifiers that implement the experiment.

    Raises:
        NoActuatorWithExperimentError: If no actuators implement the experiment.
    """
    actuators_with_target_experiment = get_actuators_implementing_experiment(
        experiment_id
    )

    # The experiment does not exist
    if len(actuators_with_target_experiment) == 0:
        raise NoActuatorWithExperimentError
    # One or more actuators implement the experiment
    return actuators_with_target_experiment


def get_actuator_from_experiment_id(
    experiment_id: str, actuator_id: str | None = None
) -> str:
    """
    Retrieves the ID of the actuator that implements an experiment ID.
    If an actuator ID is also provided, the method will validate that the
    actuator implements the experiment.

    Parameters:
    - experiment_id (str): The experiment ID to retrieve the actuator ID for.
    - actuator_id (str, optional): The actuator ID to check for the experiment. Defaults to None.

    Returns:
    str: The actuator ID that implements the experiment.

    Raises:
    NoActuatorWithExperimentError: If no actuators implement the experiment.
    ActuatorDoesNotHaveExperimentError: If the provided actuator ID does not implement the experiment.
    TooManyActuatorsWithExperimentError: If multiple actuators implement the experiment and no actuator ID is provided.
    """

    actuators_with_target_experiment = get_actuators_from_experiment_id(experiment_id)

    if len(actuators_with_target_experiment) > 1:
        # An actuator ID was provided, but it does not have the requested experiment
        if actuator_id and actuator_id not in actuators_with_target_experiment:
            raise ActuatorDoesNotHaveExperimentError(actuators_with_target_experiment)
        # We don't know which actuator the user wants the experiment from
        if not actuator_id:
            raise TooManyActuatorsWithExperimentError(actuators_with_target_experiment)
    # Only one actuator implements the experiment
    else:
        # The user-provided actuator ID does not implement the experiment
        original_actuator_id = actuator_id
        actuator_id = next(iter(actuators_with_target_experiment))
        if original_actuator_id and original_actuator_id != actuator_id:
            raise ActuatorDoesNotHaveExperimentError(actuators_with_target_experiment)

    return actuator_id


def _ado_get_actuator_from_experiment_id(
    experiment_id: str, actuator_id: str | None = None
) -> str:
    try:
        return get_actuator_from_experiment_id(experiment_id, actuator_id)
    except NoActuatorWithExperimentError as e:
        console_print(
            f"{ERROR}Experiment {magenta(experiment_id)} does not exist", stderr=True
        )
        raise typer.Exit(1) from e
    except TooManyActuatorsWithExperimentError as e:
        console_print(
            f"{ERROR}Experiment {magenta(experiment_id)} was found in "
            f"multiple actuators: {e.actuators_with_experiments}. "
            "Use the --actuator-id flag to select one.",
            stderr=True,
        )
        raise typer.Exit(1) from e
    except ActuatorDoesNotHaveExperimentError as e:
        hint_text = (
            f"{HINT}Did you mean one of {e.actuators_with_experiments}?"
            if len(e.actuators_with_experiments) > 1
            else f"{HINT}Did you mean {magenta(e.actuators_with_experiments.pop())}?"
        )
        console_print(
            f"{ERROR}Requested actuator {magenta(actuator_id)} does not have "
            f"experiment {magenta(experiment_id)}\n{hint_text}",
            stderr=True,
        )
        raise typer.Exit(1) from e
