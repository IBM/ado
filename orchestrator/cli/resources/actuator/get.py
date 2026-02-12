# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import rich.box
import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import AdoGetSupportedOutputFormats
from orchestrator.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY,
    ERROR,
    HINT,
    WARN,
    console_print,
)
from orchestrator.utilities.rich import dataframe_to_rich_table


def get_actuator(parameters: AdoGetCommandParameters) -> None:

    console_print(
        f"{WARN}These functionalities are global, and not context-aware\n"
        f"{WARN}This a local command. It will not reflect the actuators on a remote cluster.",
        stderr=True,
    )

    import pandas as pd

    import orchestrator.modules.actuators
    import orchestrator.modules.actuators.registry

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY) as spinner:
        registry = (
            orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
        )

        if (
            parameters.resource_id
            and parameters.resource_id not in registry.actuatorIdentifierMap
        ):
            spinner.stop()
            console_print(
                f"{ERROR}Actuator {parameters.resource_id} does not exist.\n"
                f"{HINT}Available actuators are: {list(registry.actuatorIdentifierMap.keys())}",
                stderr=True,
            )
            raise typer.Exit(1)

        if parameters.output_format != AdoGetSupportedOutputFormats.DEFAULT:
            spinner.stop()
            console_print(
                f"{ERROR}Only the {AdoGetSupportedOutputFormats.DEFAULT.value} output format "
                "is supported by this command.",
                stderr=True,
            )
            raise typer.Exit(1)

        spinner.update(ADO_SPINNER_GETTING_OUTPUT_READY)

        data = []
        columns = ["ACTUATOR ID", "EXPERIMENTS"]

        if parameters.show_details:
            columns.extend(["DESCRIPTION", "VERSION"])

        if parameters.resource_id:
            actuator_identifiers = [parameters.resource_id]
        else:
            actuator_identifiers = registry.actuatorIdentifierMap.keys()

        for actuator_id in sorted(actuator_identifiers):
            catalog = registry.catalogForActuatorIdentifier(actuator_id)

            # Count all experiments
            total_experiments = len(catalog.experiments)

            row = [actuator_id, total_experiments]

            if parameters.show_details:
                actuator_metadata = registry.actuatorMetadataMap.get(actuator_id, {})
                row.extend(
                    [
                        actuator_metadata.get("description") or "",
                        actuator_metadata.get("version") or "",
                    ]
                )

            data.append(row)

        output_df = pd.DataFrame(
            data=data,
            columns=columns,
        )

        if output_df.empty:
            spinner.stop()
            console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            return

    console_print(
        dataframe_to_rich_table(output_df, box=rich.box.SQUARE, show_edge=True)
    )
