# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from ado.cli.models.parameters import AdoShowStatsCommandParameters
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from ado.cli.utils.output.stats import render_stats_dataframe
from ado.cli.utils.resources.formatters import (
    build_resource_listing_dataframe,
    format_ado_get_stats_for_spaces,
    format_default_ado_get_multiple_resources,
)
from ado.core.resources import ADOResource, CoreResourceKinds


def show_discovery_space_stats(parameters: AdoShowStatsCommandParameters) -> None:
    """Show statistics for one or more discovery spaces.

    Outputs all standard ``ado get`` table columns (IDENTIFIER, NAME, AGE)
    plus lightweight stats columns (EXPERIMENTS, OPERATIONS,
    EXPLORE_OPERATIONS, MEASURED_ENTITIES — entities with at least one
    measurement, whether successful, failed, or both) and heavy stats columns
    (SIZE_OF_ENTITY_SPACE, UNMEASURED_ENTITIES, MATCHING_ENTITIES,
    MATCHING_WITH_MEASUREMENTS, ENTITIES_WITH_ALL_MEASUREMENTS,
    ENTITIES_WITH_PARTIAL_MEASUREMENTS,
    MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS).

    Args:
        parameters: Command parameters including resource IDs, output format,
            output file, query filters, and render flag.
    """
    if parameters.filters and parameters.resource_ids:
        console_print(
            f"{ERROR}You cannot specify space ids and filters/labels at the same time",
            stderr=True,
        )
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if parameters.resource_ids:
            space_resources: dict[str, ADOResource] = sql_store.getResources(
                parameters.resource_ids
            )
        else:
            space_resources = sql_store.getResourcesOfKind(
                kind=CoreResourceKinds.DISCOVERYSPACE.value,
                field_selectors=parameters.filters,
            )

        base_df = format_default_ado_get_multiple_resources(
            resources=build_resource_listing_dataframe(
                resources=space_resources,
                resource_kind=CoreResourceKinds.DISCOVERYSPACE,
                show_details=parameters.show_details,
            ),
            resource_kind=CoreResourceKinds.DISCOVERYSPACE,
        )

        if base_df.empty:
            if parameters.filters:
                console_print(
                    f"{ERROR}The filter/labels provided did not match any space.",
                    stderr=True,
                )
                raise typer.Exit(1)
            if parameters.resource_ids:
                console_print(
                    f"{ERROR}No data was retrieved for any of the spaces: {parameters.resource_ids}",
                    stderr=True,
                )
                raise typer.Exit(1)
            console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            return

        # Compute all stats (lightweight + heavy) in one pass.
        status.update(ADO_SPINNER_GETTING_OUTPUT_READY)
        df = format_ado_get_stats_for_spaces(
            base_df,
            sql_store,
            spinner=status,
            include_heavy=True,
            space_resources=space_resources,
            project_context=parameters.ado_configuration.project_context,
        )

    render_stats_dataframe(
        df=df,
        output_format=parameters.output_format,
        output_file=parameters.output_file,
        render_output=parameters.render_output,
    )
