# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import AdoGetSupportedOutputFormats
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY,
    ERROR,
    HINT,
    INFO,
    console_print,
)


def get_actuator(parameters: AdoGetCommandParameters) -> None:

    if not parameters.no_trunc:
        parameters.no_trunc = ["ACTUATOR ID"]

    console_print(
        f"{INFO}This is a local command. It will not reflect the actuators on a remote cluster.",
        stderr=True,
    )

    # Validate output format early
    if parameters.output_format not in {
        AdoGetSupportedOutputFormats.TABLE,
        AdoGetSupportedOutputFormats.NAME,
    }:
        console_print(
            f"{ERROR}Only the {AdoGetSupportedOutputFormats.TABLE.value} and "
            f"{AdoGetSupportedOutputFormats.NAME.value} output formats "
            "are supported by this command.",
            stderr=True,
        )
        raise typer.Exit(1)

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

        spinner.stop()

    from orchestrator.cli.utils.resources.handlers import handle_ado_get

    # Use unified handler for rendering
    handle_ado_get(parameters=parameters, dataframe=output_df)
