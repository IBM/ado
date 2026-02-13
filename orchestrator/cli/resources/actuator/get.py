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
    INFO,
    console_print,
)
from orchestrator.utilities.rich import dataframe_to_rich_table


def get_actuator(parameters: AdoGetCommandParameters) -> None:

    console_print(
        f"{INFO}This is a local command. It will not reflect the actuators on a remote cluster.",
        stderr=True,
    )

    import pandas as pd

    from orchestrator.modules.actuators.registry import ActuatorRegistry

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY) as spinner:
        registry = ActuatorRegistry.globalRegistry()
        available_actuators = sorted(registry.actuatorIdentifierMap.keys())

        # Validate actuator exists if specific ID provided
        if (
            parameters.resource_id
            and parameters.resource_id not in registry.actuatorIdentifierMap
        ):
            spinner.stop()
            console_print(
                f"{ERROR}Actuator '{parameters.resource_id}' does not exist.\n"
                f"{HINT}Available actuators are: {available_actuators}",
                stderr=True,
            )
            raise typer.Exit(1)

        # Validate output format
        if parameters.output_format != AdoGetSupportedOutputFormats.DEFAULT:
            spinner.stop()
            console_print(
                f"{ERROR}Only the {AdoGetSupportedOutputFormats.DEFAULT.value} output format "
                "is supported by this command.",
                stderr=True,
            )
            raise typer.Exit(1)

        spinner.update(ADO_SPINNER_GETTING_OUTPUT_READY)

        # Build column structure
        columns = ["ACTUATOR ID", "EXPERIMENTS"]
        if parameters.show_details:
            columns.extend(["DESCRIPTION", "VERSION"])

        # Determine which actuators to display
        actuator_identifiers = (
            [parameters.resource_id] if parameters.resource_id else available_actuators
        )

        # Collect actuator data
        data = []
        for actuator_id in actuator_identifiers:
            catalog = registry.catalogForActuatorIdentifier(actuator_id)
            total_experiments = len(catalog.experiments)

            row = [actuator_id, total_experiments]

            if parameters.show_details:
                actuator_metadata = registry.actuatorMetadataMap.get(actuator_id, {})
                row.extend(
                    [
                        actuator_metadata.get("description", ""),
                        actuator_metadata.get("version", ""),
                    ]
                )

            data.append(row)

        # Create DataFrame
        output_df = pd.DataFrame(data=data, columns=columns)

        if output_df.empty:
            spinner.stop()
            console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            return

    console_print(
        dataframe_to_rich_table(output_df, box=rich.box.SQUARE, show_edge=True)
    )
