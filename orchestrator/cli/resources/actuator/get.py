# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import AdoGetSupportedOutputFormats
from orchestrator.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY,
    ERROR,
    HINT,
    INFO,
    WARN,
    console_print,
)


def get_actuator(parameters: AdoGetCommandParameters) -> None:

    console_print(
        f"{WARN}These functionalities are global, and not context-aware\n"
        f"{WARN}This a local command. It will not reflect the actuators on a remote cluster.",
        stderr=True,
    )

    import pandas as pd

    import orchestrator.modules.actuators
    import orchestrator.modules.actuators.registry

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY):
        registry = (
            orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
        )

    if (
        parameters.resource_id
        and parameters.resource_id not in registry.actuatorIdentifierMap
    ):
        console_print(
            f"{ERROR}Actuator {parameters.resource_id} does not exist.\n"
            f"{HINT}Available actuators are: {list(registry.actuatorIdentifierMap.keys())}",
            stderr=True,
        )
        raise typer.Exit(1)

    if parameters.output_format != AdoGetSupportedOutputFormats.DEFAULT:
        console_print(
            f"{ERROR}Only the {AdoGetSupportedOutputFormats.DEFAULT.value} output format "
            "is supported by this command.",
            stderr=True,
        )
        raise typer.Exit(1)

    if not parameters.show_details:
        data = list(registry.actuatorIdentifierMap.keys())
        columns = ["ACTUATOR ID"]
    else:
        data = []
        columns = ["ACTUATOR ID", "CATALOG ID", "EXPERIMENT ID", "SUPPORTED"]

        if parameters.resource_id:
            actuator_identifiers = [parameters.resource_id]
        else:
            actuator_identifiers = registry.actuatorIdentifierMap.keys()

        for actuator_id in actuator_identifiers:
            catalog = registry.catalogForActuatorIdentifier(actuator_id)
            if not catalog.experiments:
                console_print(
                    f"{INFO}Actuator {actuator_id} has been omitted as it does not provide any experiment.",
                    stderr=True,
                )
                continue

            data.extend(
                [
                    actuator_id,
                    catalog.identifier,
                    experiment.identifier,
                    not experiment.deprecated,
                ]
                for experiment in catalog.experiments
                if not experiment.deprecated or parameters.show_deprecated
            )

    output_df = pd.DataFrame(
        data=data,
        columns=columns,
    )

    if parameters.resource_id:
        output_df = output_df[output_df["ACTUATOR ID"] == parameters.resource_id]
        output_df = output_df.reset_index(drop=True)

    if output_df.empty:
        console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
        return

    console_print(output_df, has_pandas_content=True)
