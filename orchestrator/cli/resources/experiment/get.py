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
    WARN,
    console_print,
)
from orchestrator.utilities.rich import dataframe_to_rich_table


def get_experiment(parameters: AdoGetCommandParameters) -> None:
    """
    List experiments and their actuators.

    Basic mode: Shows EXPERIMENT ID and ACTUATOR ID columns
    Details mode: Adds DESCRIPTION and DEPRECATED columns
    """

    console_print(
        f"{WARN}This is a local command. It will not reflect the experiments on a remote cluster.",
        stderr=True,
    )

    import pandas as pd

    import orchestrator.modules.actuators
    import orchestrator.modules.actuators.registry

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY):
        registry = (
            orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
        )

    # Validate output format
    if parameters.output_format != AdoGetSupportedOutputFormats.DEFAULT:
        console_print(
            f"{ERROR}Only the {AdoGetSupportedOutputFormats.DEFAULT.value} output format "
            "is supported by this command.",
            stderr=True,
        )
        raise typer.Exit(1)

    # Collect experiment data
    data = []

    if not parameters.show_details:
        columns = ["EXPERIMENT ID", "ACTUATOR ID"]
    else:
        columns = [
            "EXPERIMENT ID",
            "ACTUATOR ID",
            "DESCRIPTION",
            "DEPRECATED",
        ]

    # Iterate through all actuators and their experiments
    for actuator_id in sorted(registry.actuatorIdentifierMap.keys()):
        catalog = registry.catalogForActuatorIdentifier(actuator_id)

        for experiment in catalog.experiments:
            # Skip deprecated experiments unless explicitly requested
            if experiment.deprecated and not parameters.show_deprecated:
                continue

            # Filter by specific experiment ID if provided
            if (
                parameters.resource_id
                and experiment.identifier != parameters.resource_id
            ):
                continue

            if not parameters.show_details:
                data.append(
                    [
                        experiment.identifier,
                        actuator_id,
                    ]
                )
            else:
                # Extract description from metadata if available
                description = experiment.metadata.get("description", "")

                data.append(
                    [
                        experiment.identifier,
                        actuator_id,
                        description,
                        experiment.deprecated,
                    ]
                )

    # Create DataFrame
    output_df = pd.DataFrame(data=data, columns=columns)

    # Check if we found the requested experiment
    if parameters.resource_id and output_df.empty:
        console_print(
            f"{ERROR}Experiment {parameters.resource_id} does not exist.",
            stderr=True,
        )
        raise typer.Exit(1)

    if output_df.empty:
        console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
        return

    # Sort by experiment ID (primary) and actuator ID (secondary)
    output_df = output_df.sort_values(
        by=["EXPERIMENT ID", "ACTUATOR ID"], ignore_index=True
    )

    console_print(dataframe_to_rich_table(output_df))
