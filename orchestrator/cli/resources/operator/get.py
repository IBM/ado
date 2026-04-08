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

    # Validate output format
    if parameters.output_format not in {
        AdoGetSupportedOutputFormats.DEFAULT,
        AdoGetSupportedOutputFormats.NAME,
    }:
        console_print(
            f"{WARN}{cyan('ado get operators')} only supports the "
            f"{AdoGetSupportedOutputFormats.DEFAULT.value} and "
            f"{AdoGetSupportedOutputFormats.NAME.value} output formats",
            stderr=True,
        )
        parameters.output_format = AdoGetSupportedOutputFormats.DEFAULT

    # Handle NAME output format
    if parameters.output_format == AdoGetSupportedOutputFormats.NAME:

        # Collect all operator names
        operator_names = []
        for (
            collection
        ) in orchestrator.modules.operators.collections.operationCollectionMap.values():
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

    # Build entries for DEFAULT format
    entries = []
    for (
        collection
    ) in orchestrator.modules.operators.collections.operationCollectionMap.values():
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

    # AP: We want to rename some DiscoveryOperationEnums
    type_names_mapping = {"search": "explore"}
    operators["TYPE"] = operators["TYPE"].replace(type_names_mapping)

    # After renaming some entries in the TYPE column
    # the values may not be sorted anymore
    operators = operators.sort_values(by=["TYPE", "OPERATOR"]).reset_index(drop=True)
    console_print(
        dataframe_to_rich_table(
            operators,
            show_edge=True,
            show_index=True,
            box=rich.box.SQUARE,
            do_not_truncate_column_content=parameters.no_trunc,
        )
    )
