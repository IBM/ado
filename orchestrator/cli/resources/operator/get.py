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
    ERROR,
    HINT,
    WARN,
    console_print,
    cyan,
)
from orchestrator.utilities.rich import dataframe_to_rich_table
from orchestrator.utilities.strings import (
    normalize_and_truncate_at_period,
)


def get_operator(parameters: AdoGetCommandParameters) -> None:

    with Status(ADO_SPINNER_GETTING_OUTPUT_READY):
        import pandas as pd

        import orchestrator.modules.operators.collections

    if parameters.output_format != AdoGetSupportedOutputFormats.DEFAULT:
        console_print(
            f"{WARN}{cyan('ado get operators')} only supports the "
            f"{AdoGetSupportedOutputFormats.DEFAULT.value} output format",
            stderr=True,
        )
        parameters.output_format = AdoGetSupportedOutputFormats.DEFAULT

    entries = []
    for (
        collection
    ) in orchestrator.modules.operators.collections.operationCollectionMap.values():
        for function_name in collection.function_operations:
            entry = {
                "OPERATOR": function_name,
                "TYPE": collection.type.value,
            }
            if parameters.show_details:
                entry["DESCRIPTION"] = normalize_and_truncate_at_period(
                    collection.function_operation_descriptions[function_name]
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

    # AP: We want to rename some DiscoveryOperationEnums
    type_names_mapping = {"search": "explore"}
    operators["TYPE"] = operators["TYPE"].replace(type_names_mapping)

    # After renaming some entries in the TYPE column
    # the values may not be sorted anymore
    operators = operators.sort_values(by=["TYPE", "OPERATOR"]).reset_index(drop=True)
    console_print(
        dataframe_to_rich_table(
            operators, show_edge=True, show_index=True, box=rich.box.SQUARE
        )
    )
