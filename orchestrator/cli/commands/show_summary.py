# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from orchestrator.cli.models.parameters import AdoShowSummaryCommandParameters
from orchestrator.cli.models.types import (
    AdoShowSummarySupportedOutputFormats,
    AdoShowSummarySupportedResourceTypes,
)
from orchestrator.cli.resources.discovery_space.show_summary import (
    show_discovery_space_summary,
)
from orchestrator.cli.utils.input.parsers import (
    enum_choice_with_plural_parser,
    parse_key_value_pairs,
)
from orchestrator.cli.utils.output.prints import (
    ERROR,
    console_print,
)
from orchestrator.cli.utils.queries.parser import (
    prepare_query_filters_for_db,
)
from orchestrator.core.samplestore.base import (
    FailedToDecodeStoredEntityError,
    FailedToDecodeStoredMeasurementResultForEntityError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration

TABLE_ONLY_OPTIONS = "Table-only Options"


def show_summary_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowSummarySupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show a summary for.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoShowSummarySupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoShowSummarySupportedResourceTypes)}]",
        ),
    ],
    ids: Annotated[
        list[str] | None,
        typer.Argument(
            ...,
            help="The ids of the resources to show a summary for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Show summary for the latest identifier of the selected resource type. "
            "Ignored if resource identifiers are also specified.",
            show_default=False,
        ),
    ] = False,
    query: Annotated[
        list[str] | None,
        typer.Option(
            "--query",
            "-q",
            help="""
            Filter results by values contained in the resources. Will return all resources that match
            the input. Can be specified multiple times to ensure all filters are matched.

            Inputs must be specified in the form of key=value, where key is a path in the resource
            and value is a JSON document.

            Please refer to the documentation provided for more information and examples:
            https://ibm.github.io/ado/getting-started/ado/#using-the-field-level-querying-functionality
            """,
            show_default=False,
        ),
    ] = None,
    labels: Annotated[
        list[str] | None,
        typer.Option(
            "--label",
            "-l",
            help="""
            Filter results by labels contained in the resources' metadata.
            Can be specified multiple times.

            Labels need to be specified in the form of key=value.
            """,
            show_default=False,
        ),
    ] = None,
    include_properties: Annotated[
        list[str] | None,
        typer.Option(
            "--with-property",
            "-p",
            help="Add constitutive properties to the output. Can be specified multiple times.",
            show_default=False,
        ),
    ] = None,
    columns_to_hide: Annotated[
        list[str] | None,
        typer.Option(
            "--hide",
            help="Hide certain columns from the output. The following values can be used to hide default columns:"
            " id (Space ID);"
            " experiment (Experiments);"
            " matching (Matching and Measured);"
            " sampled (Sampled and Measured);"
            " name (Name);"
            " description (Description);",
            show_default=False,
            rich_help_panel=TABLE_ONLY_OPTIONS,
        ),
    ] = None,
    output_format: Annotated[
        AdoShowSummarySupportedOutputFormats,
        typer.Option(
            "--output",
            "-o",
            help="The format in which to output the summary. "
            "Options: table (rich console table), md-table (markdown table), "
            "md-report (markdown prose report), csv (CSV format).",
        ),
    ] = AdoShowSummarySupportedOutputFormats.TABLE.value,
    output_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output-file",
            help="Write output to the specified file instead of stdout.",
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
            show_default=False,
        ),
    ] = None,
    render_output: Annotated[
        bool,
        typer.Option(
            "--render",
            help="Render the output in the console. Only supported for markdown table and markdown report output.",
        ),
    ] = False,
) -> None:
    """
    Show a formatted summary of one or more discovery spaces.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-summary
    for detailed documentation and examples.

    Examples:

    # Show a high-level summary of the discovery space as a rich table
    ado show summary space <space-id>

    # Show a high-level summary of the latest discovery space as a rich table
    ado show summary space --use-latest

    # Show a high-level summary of discovery spaces matching a label
    ado show summary space -l key=value

    # Show a detailed summary of the discovery space as a Markdown report
    ado show summary space <space-id> -o md-report
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if use_latest:
        from orchestrator.cli.utils.generic.common import get_effective_resource_id

        # Handle single ID case - get_effective_resource_id handles precedence
        resource_id = get_effective_resource_id(
            explicit_resource_id=ids[0] if ids else None,
            resource_type=resource_type.value,
            project_context=ado_configuration.project_context,
        )
        ids = [resource_id]

    try:
        query = prepare_query_filters_for_db(parse_key_value_pairs(query))
        if labels:
            for parsed_label in parse_key_value_pairs(labels):
                for k, v in parsed_label.items():
                    query.extend(
                        prepare_query_filters_for_db({"config.metadata.labels": {k: v}})
                    )
    except ValueError as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(1) from e

    parameters = AdoShowSummaryCommandParameters(
        ado_configuration=ado_configuration,
        columns_to_hide=columns_to_hide,
        include_properties=include_properties,
        output_file=output_file,
        output_format=output_format,
        query=query,
        render_output=render_output,
        resource_ids=ids,
    )

    method_mapping = {
        AdoShowSummarySupportedResourceTypes.DISCOVERY_SPACE: show_discovery_space_summary
    }

    try:
        method_mapping[resource_type](parameters=parameters)
    except (
        FailedToDecodeStoredEntityError,
        FailedToDecodeStoredMeasurementResultForEntityError,
    ) as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(1) from e


def register_show_summary_command(app: typer.Typer) -> None:
    app.command(
        name="summary",
        no_args_is_help=True,
    )(show_summary_for_resources)
