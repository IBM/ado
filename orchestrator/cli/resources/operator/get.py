# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

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
        entries.extend(
            [
                {"OPERATOR": function_name, "TYPE": collection.type.value}
                for function_name in collection.function_operations
            ]
        )

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
    operators = operators.sort_values(by=["TYPE", "OPERATOR"])
    console_print(operators, has_pandas_content=True)
