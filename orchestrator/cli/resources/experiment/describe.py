# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoDescribeCommandParameters
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY,
    ERROR,
    HINT,
    console_print,
)
from orchestrator.cli.utils.resources.experiments import (
    _ado_get_actuator_from_experiment_id,
)
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.reference import ExperimentReference


def describe_experiment(parameters: AdoDescribeCommandParameters) -> None:

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY):
        registry = ActuatorRegistry.globalRegistry()

    if (
        parameters.actuator_id
        and parameters.actuator_id not in registry.actuatorIdentifierMap
    ):
        console_print(
            f"{ERROR}Actuator {parameters.actuator_id} does not exist.\n"
            f"{HINT}Available ones are: {list(registry.actuatorIdentifierMap.keys())}",
            stderr=True,
        )
        raise typer.Exit(1)

    #
    actuator_id = (
        parameters.actuator_id
        if parameters.actuator_id
        else _ado_get_actuator_from_experiment_id(
            experiment_id=parameters.resource_id, actuator_id=None
        )
    )
    experiment = registry.experimentForReference(
        ExperimentReference(
            experimentIdentifier=parameters.resource_id, actuatorIdentifier=actuator_id
        )
    )

    console_print(experiment)
