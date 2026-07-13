# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from rich.status import Status

from ado.cli.models.parameters import AdoDescribeCommandParameters
from ado.cli.utils.output.prints import (
    ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY,
    console_print,
)
from ado.cli.utils.resources.experiments import (
    _ado_lookup_cli_experiment,
)
from ado.modules.actuators.registry import (
    ActuatorRegistry,
)


def describe_experiment(parameters: AdoDescribeCommandParameters) -> None:

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY):
        registry = ActuatorRegistry.globalRegistry()
        experiment = _ado_lookup_cli_experiment(
            parameters.resource_id, registry=registry
        )

    console_print(experiment)
