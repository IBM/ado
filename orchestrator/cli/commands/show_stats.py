# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from orchestrator.cli.models.parameters import AdoShowStatsCommandParameters
from orchestrator.cli.models.types import (
    AdoShowStatsSupportedOutputFormats,
    AdoShowStatsSupportedResourceTypes,
)
from orchestrator.cli.resources.datacontainer.show_stats import show_datacontainer_stats
from orchestrator.cli.resources.discovery_space.show_stats import (
    show_discovery_space_stats,
)
from orchestrator.cli.resources.operation.show_stats import show_operation_stats
from orchestrator.cli.resources.samplestore.show_stats import show_samplestore_stats
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

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


def show_stats_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowStatsSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show full statistics for.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoShowStatsSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoShowStatsSupportedResourceTypes)}]",
        ),
    ],
    ids: Annotated[
        list[str] | None,
        typer.Argument(
            ...,
            help="The ids of the resources to show statistics for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Show stats for the latest identifier of the selected resource type. "
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
    output_format: Annotated[
        AdoShowStatsSupportedOutputFormats,
        typer.Option(
            "--output",
            "-o",
            help="The format in which to output the statistics. "
            "Options: table (rich console table), md-table (markdown table), "
            "csv (CSV format), json (JSON), yaml (YAML).",
        ),
    ] = AdoShowStatsSupportedOutputFormats.TABLE.value,
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
    show_details: Annotated[
        bool,
        typer.Option(
            "--details",
            help="Output additional information on each object, such as names and descriptions.",
            show_default=True,
        ),
    ] = False,
    render_output: Annotated[
        bool,
        typer.Option(
            "--render",
            help="Render the output in the console. Only supported for markdown table output.",
        ),
    ] = False,
) -> None:
    """
    Show full in-depth statistics for one or more resources.

    Examples:

    # Show stats for all operations
    ado show stats operation

    # Show stats for a specific discovery space
    ado show stats discoveryspace <space-id>

    # Show stats for the latest operation as JSON
    ado show stats operation --use-latest -o json

    # Show stats for samplstores matching a label
    ado show stats samplestore -l key=value
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if use_latest:
        from orchestrator.cli.utils.generic.common import get_effective_resource_id

        resource_id = get_effective_resource_id(
            explicit_resource_id=ids[0] if ids else None,
            resource_type=resource_type.value,
            project_context=ado_configuration.project_context,
        )
        ids = [resource_id]

    try:
        query_filters = prepare_query_filters_for_db(parse_key_value_pairs(query))
        if labels:
            for parsed_label in parse_key_value_pairs(labels):
                for k, v in parsed_label.items():
                    query_filters.extend(
                        prepare_query_filters_for_db({"config.metadata.labels": {k: v}})
                    )
    except ValueError as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(1) from e

    parameters = AdoShowStatsCommandParameters(
        ado_configuration=ado_configuration,
        output_format=output_format,
        output_file=output_file,
        query=query_filters if (query or labels) else None,
        render_output=render_output,
        resource_ids=ids,
        show_details=show_details,
    )

    method_mapping = {
        AdoShowStatsSupportedResourceTypes.DISCOVERY_SPACE: show_discovery_space_stats,
        AdoShowStatsSupportedResourceTypes.OPERATION: show_operation_stats,
        AdoShowStatsSupportedResourceTypes.SAMPLE_STORE: show_samplestore_stats,
        AdoShowStatsSupportedResourceTypes.DATA_CONTAINER: show_datacontainer_stats,
    }

    method_mapping[resource_type](parameters=parameters)


def register_show_stats_command(app: typer.Typer) -> None:
    """Register the 'stats' subcommand onto the given typer app.

    Args:
        app: The typer application to register the command on.
    """
    app.command(
        name="stats",
        no_args_is_help=True,
    )(show_stats_for_resources)
