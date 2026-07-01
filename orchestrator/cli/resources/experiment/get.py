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
    WARN,
    console_print,
)
from orchestrator.cli.utils.resources.experiments import (
    _ado_experiment_from_cli_resource_id,
)


def get_experiment(parameters: AdoGetCommandParameters) -> None:
    """
    List experiments and their actuators.

    Basic mode: Shows ACTUATOR ID, EXPERIMENT ID, and VERSION columns
    Details mode: Adds DESCRIPTION and DEPRECATED columns
    """

    if not parameters.no_trunc:
        parameters.no_trunc = ["EXPERIMENT ID"]

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

        if parameters.output_format != AdoGetSupportedOutputFormats.TABLE:
            spinner.stop()
            console_print(
                f"{ERROR}Only the {AdoGetSupportedOutputFormats.TABLE.value} output format "
                "is supported by this command.",
                stderr=True,
            )
            raise typer.Exit(1)

        spinner.update(ADO_SPINNER_GETTING_OUTPUT_READY)
        data = []

        if not parameters.show_details:
            columns = ["ACTUATOR ID", "EXPERIMENT ID", "VERSION"]
        else:
            columns = [
                "ACTUATOR ID",
                "EXPERIMENT ID",
                "VERSION",
                "DESCRIPTION",
            ]

        if parameters.show_deprecated:
            columns.append("SUPPORTED")

        if parameters.resource_id:
            matched_experiment = _ado_experiment_from_cli_resource_id(
                parameters.resource_id,
                registry=registry,
            )
            if matched_experiment.deprecated and not parameters.show_deprecated:
                experiments_to_show = []
            else:
                experiments_to_show = [
                    (matched_experiment.actuatorIdentifier, matched_experiment)
                ]
        else:
            experiments_to_show = []
            for actuator_id in sorted(registry.actuatorIdentifierMap.keys()):
                catalog = registry.catalogForActuatorIdentifier(actuator_id)
                for experiment in catalog.experiments:
                    if experiment.deprecated and not parameters.show_deprecated:
                        continue
                    experiments_to_show.append((actuator_id, experiment))

        for actuator_id, experiment in experiments_to_show:
            row = [
                actuator_id,
                experiment.identifier,
                experiment.version,
            ]

            if parameters.show_details:
                row.append(experiment.metadata.get("description", ""))

            if parameters.show_deprecated:
                row.append(not experiment.deprecated)

            data.append(row)

        output_df = pd.DataFrame(data=data, columns=columns)

        if parameters.resource_id and output_df.empty:
            spinner.stop()
            console_print(
                f"{ERROR}Experiment {parameters.resource_id} does not exist.",
                stderr=True,
            )
            raise typer.Exit(1)

        if not output_df.empty:
            output_df = output_df.sort_values(
                by=["ACTUATOR ID", "EXPERIMENT ID"], ignore_index=True
            )

        spinner.stop()

    from orchestrator.cli.utils.resources.handlers import handle_ado_get

    handle_ado_get(parameters=parameters, dataframe=output_df)
