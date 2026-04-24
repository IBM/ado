# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowSummaryCommandParameters
from orchestrator.cli.models.types import (
    AdoShowSummarySupportedOutputFormats,
)
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    SUCCESS,
    WARN,
    console_print,
    magenta,
)
from orchestrator.core.resources import CoreResourceKinds


def show_discovery_space_summary(parameters: AdoShowSummaryCommandParameters) -> None:
    import pandas as pd

    if parameters.query and parameters.resource_ids:
        console_print(
            f"{ERROR}You cannot specify space ids and queries/labels at the same time",
            stderr=True,
        )
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:

        if parameters.query:
            spaces = sql_store.getResourceIdentifiersOfKind(
                kind=CoreResourceKinds.DISCOVERYSPACE.value,
                field_selectors=parameters.query,
            )

            if spaces.empty:
                console_print(
                    f"{ERROR}The query/labels provided did not match any space.",
                    stderr=True,
                )
                raise typer.Exit(1)

            parameters.resource_ids = list(spaces["IDENTIFIER"])

        # Load from DB
        space_resources = sql_store.getResources(identifiers=parameters.resource_ids)

        # Time-consuming imports
        status.update("Preparing to create your summary")

        from orchestrator.cli.models.space import SpaceSummary

        summaries: list[SpaceSummary] = []
        for space_id in space_resources:
            status.update(
                f"Preparing summary for space {magenta(space_id)} ({len(summaries) + 1}/{len(space_resources)})"
            )
            summaries.append(
                SpaceSummary(space_id, parameters.ado_configuration.project_context)
            )

        if not summaries:
            console_print(
                f"{ERROR}No data was retrieved for any of the spaces: {parameters.resource_ids}",
                stderr=True,
            )
            raise typer.Exit(1)

        if (
            parameters.output_format
            == AdoShowSummarySupportedOutputFormats.MARKDOWN_REPORT
        ):

            if parameters.include_properties:
                console_print(
                    f"{WARN}It's not possible to restrict the constitutive properties shown "
                    f"when using {AdoShowSummarySupportedOutputFormats.MARKDOWN_REPORT.value} output.",
                    stderr=True,
                )

            result = "\n".join([summary.to_markdown_text() for summary in summaries])

        else:

            df = pd.concat(
                [
                    summary.to_dataframe(
                        include_properties=parameters.include_properties,
                        columns_to_hide=parameters.columns_to_hide,
                    )
                    for summary in summaries
                ]
            ).fillna("")

            if parameters.output_format == AdoShowSummarySupportedOutputFormats.TABLE:

                import rich.box

                from orchestrator.utilities.rich import (
                    dataframe_to_rich_table,
                    render_to_string,
                )

                # When writing to file, avoid truncating columns by default
                table = dataframe_to_rich_table(
                    df,
                    show_edge=True,
                    show_index=True,
                    box=rich.box.SQUARE,
                    do_not_truncate_columns=parameters.output_file is not None,
                )
                result = render_to_string(table, auto_width=True)

            if parameters.output_format == AdoShowSummarySupportedOutputFormats.CSV:
                result = df.to_csv()
            elif (
                parameters.output_format
                == AdoShowSummarySupportedOutputFormats.MARKDOWN_TABLE
            ):
                result = df.to_markdown()

        if parameters.output_file:
            parameters.output_file.write_text(result)
            console_print(
                f"{SUCCESS} Output saved as {magenta(str(parameters.output_file))}",
                stderr=True,
            )
            return

        if parameters.render_output and parameters.output_format in {
            AdoShowSummarySupportedOutputFormats.MARKDOWN_REPORT,
            AdoShowSummarySupportedOutputFormats.MARKDOWN_TABLE,
        }:
            from rich.markdown import Markdown

            result = Markdown(result)

    console_print(result)
