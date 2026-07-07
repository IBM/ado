# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import typer
from rich.status import Status

from ado.cli.models.parameters import AdoGetCommandParameters
from ado.cli.models.types import AdoGetSupportedOutputFormats
from ado.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ERROR,
    HINT,
    WARN,
    console_print,
    cyan,
)
from ado.utilities.strings import (
    normalize_and_truncate_at_period,
)


def get_operator(parameters: AdoGetCommandParameters) -> None:

    if not parameters.no_trunc:
        parameters.no_trunc = ["OPERATOR"]

    with Status(ADO_SPINNER_GETTING_OUTPUT_READY):
        import pandas as pd

        import ado.modules.operators.collections

    # Validate output format
    if parameters.output_format not in {
        AdoGetSupportedOutputFormats.TABLE,
        AdoGetSupportedOutputFormats.NAME,
    }:
        console_print(
            f"{WARN}{cyan('ado get operators')} only supports the "
            f"{AdoGetSupportedOutputFormats.TABLE.value} and "
            f"{AdoGetSupportedOutputFormats.NAME.value} output formats",
            stderr=True,
        )
        parameters.output_format = AdoGetSupportedOutputFormats.TABLE

    # Handle NAME output format
    if parameters.output_format == AdoGetSupportedOutputFormats.NAME:
        # Collect all operator names
        operator_names = []
        for (
            collection
        ) in ado.modules.operators.collections.operationCollectionMap.values():
            operator_names.extend(collection.operators.keys())

        if parameters.resource_id:
            # Single operator: verify it exists and output its name
            if parameters.resource_id not in operator_names:
                console_print(
                    f"{ERROR}{parameters.resource_id} is not among the available operators.\n"
                    f"{HINT}Run {cyan('ado get operators')} to list them.",
                    stderr=True,
                )
                raise typer.Exit(1)
            console_print(parameters.resource_id)
        else:
            # Multiple operators: output all names
            for operator_name in sorted(operator_names):
                console_print(operator_name)
        return

    # Build entries for TABLE format
    entries = []
    for collection in ado.modules.operators.collections.operationCollectionMap.values():
        for operator_name, operator in collection.operators.items():
            entry = {
                "OPERATOR": operator_name,
                "VERSION": operator.version,
                "TYPE": collection.type.value,
            }
            if parameters.show_details:
                entry["DESCRIPTION"] = normalize_and_truncate_at_period(
                    operator.description or ""
                )
            entries.append(entry)

    operators = pd.DataFrame(entries)
    if operators.empty:
        console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
        return

    if parameters.resource_id:
        operators = operators[operators["OPERATOR"] == parameters.resource_id]
        operators = operators.reset_index(drop=True)

        if operators.empty:
            console_print(
                f"{ERROR}{parameters.resource_id} is not among the available operators.\n"
                f"{HINT}Run {cyan('ado get operators')} to list them.",
                stderr=True,
            )
            raise typer.Exit(1)
    else:
        console_print("Available operators by type:")

    operators = operators.sort_values(by=["TYPE", "OPERATOR"]).reset_index(drop=True)

    from ado.cli.utils.resources.handlers import handle_ado_get

    # Use unified handler for rendering
    handle_ado_get(parameters=parameters, dataframe=operators)
