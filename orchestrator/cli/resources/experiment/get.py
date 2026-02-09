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

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY) as spinner:
        registry = (
            orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
        )

        # Validate output format
        if parameters.output_format != AdoGetSupportedOutputFormats.DEFAULT:
            spinner.stop()
            console_print(
                f"{ERROR}Only the {AdoGetSupportedOutputFormats.DEFAULT.value} output format "
                "is supported by this command.",
                stderr=True,
            )
            raise typer.Exit(1)

        # Collect experiment data
        spinner.update(ADO_SPINNER_GETTING_OUTPUT_READY)
        data = []

        if not parameters.show_details:
            columns = ["ACTUATOR ID", "EXPERIMENT ID"]
        else:
            columns = [
                "ACTUATOR ID",
                "EXPERIMENT ID",
                "DESCRIPTION",
            ]

        if parameters.show_deprecated:
            columns.append("SUPPORTED")

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

                # Have Actuator ID and Experiment ID by default
                row = [
                    actuator_id,
                    experiment.identifier,
                ]

                # Show details adds description
                if parameters.show_details:
                    row.append(experiment.metadata.get("description", ""))

                # Show deprecated requires adding the supported column
                if parameters.show_deprecated:
                    row.append(not experiment.deprecated)

                data.append(row)

        # Create DataFrame
        output_df = pd.DataFrame(data=data, columns=columns)

        # Check if we found the requested experiment
        if parameters.resource_id and output_df.empty:
            spinner.stop()
            console_print(
                f"{ERROR}Experiment {parameters.resource_id} does not exist.",
                stderr=True,
            )
            raise typer.Exit(1)

        if output_df.empty:
            spinner.stop()
            console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            return

        # Sort by actuator ID (primary) and experiment ID (secondary)
        output_df = output_df.sort_values(
            by=["ACTUATOR ID", "EXPERIMENT ID"], ignore_index=True
        )

    console_print(
        dataframe_to_rich_table(output_df, box=rich.box.SQUARE, show_edge=True)
    )
