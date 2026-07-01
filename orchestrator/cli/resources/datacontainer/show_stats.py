# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowStatsCommandParameters
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from orchestrator.cli.utils.output.stats import render_stats_dataframe
from orchestrator.cli.utils.resources.formatters import (
    build_resource_listing_dataframe,
    format_ado_get_stats_for_datacontainers,
    format_default_ado_get_multiple_resources,
)
from orchestrator.core.resources import CoreResourceKinds


def show_datacontainer_stats(parameters: AdoShowStatsCommandParameters) -> None:
    """Show statistics for one or more data containers.

    Outputs all standard ``ado get`` table columns (IDENTIFIER, NAME, AGE)
    plus TABLES, LOCATIONS, KEY_VALUES, and DATA_BYTES stats columns.

    Args:
        parameters: Command parameters including resource IDs, output format,
            output file, query filters, and render flag.
    """
    if parameters.filters and parameters.resource_ids:
        console_print(
            f"{ERROR}You cannot specify datacontainer ids and filters/labels at the same time",
            stderr=True,
        )
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if parameters.resource_ids:
            datacontainer_resources = sql_store.getResources(parameters.resource_ids)
            base_df = format_default_ado_get_multiple_resources(
                resources=build_resource_listing_dataframe(
                    resources=datacontainer_resources,
                    resource_kind=CoreResourceKinds.DATACONTAINER,
                    show_details=parameters.show_details,
                ),
                resource_kind=CoreResourceKinds.DATACONTAINER,
            )
        else:
            datacontainer_resources = sql_store.getResourceIdentifiersOfKind(
                kind=CoreResourceKinds.DATACONTAINER.value,
                field_selectors=parameters.filters,
                details=parameters.show_details,
            )
            base_df = format_default_ado_get_multiple_resources(
                resources=datacontainer_resources,
                resource_kind=CoreResourceKinds.DATACONTAINER,
            )

        if base_df.empty:
            if parameters.filters:
                console_print(
                    f"{ERROR}The filter/labels provided did not match any datacontainer.",
                    stderr=True,
                )
                raise typer.Exit(1)
            if parameters.resource_ids:
                console_print(
                    f"{ERROR}No data was retrieved for any of the datacontainers: {parameters.resource_ids}",
                    stderr=True,
                )
                raise typer.Exit(1)
            console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            return

        status.update(ADO_SPINNER_GETTING_OUTPUT_READY)
        df = format_ado_get_stats_for_datacontainers(base_df, sql_store, spinner=status)

    render_stats_dataframe(
        df=df,
        output_format=parameters.output_format,
        output_file=parameters.output_file,
        render_output=parameters.render_output,
    )
