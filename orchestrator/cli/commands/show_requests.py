# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoShowRequestsCommandParameters
from orchestrator.cli.models.types import (
    AdoShowRequestsSupportedOutputFormats,
    AdoShowRequestsSupportedResourceTypes,
)
from orchestrator.cli.resources.operation.show_requests import show_operation_requests
from orchestrator.cli.utils.generic.common import get_effective_resource_id
from orchestrator.cli.utils.output.prints import ERROR, console_print

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


def show_requests_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowRequestsSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show the request timeseries for.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoShowRequestsSupportedResourceTypes),
        ),
    ],
    resource_id: Annotated[
        str | None,
        typer.Argument(
            ...,
            help="The id of the resource to show the request timeseries for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Show the timeseries of requests for the latest identifier of the selected resource type. "
            "Ignored if a resource identifier is also specified.",
            show_default=False,
        ),
    ] = False,
    output_format: Annotated[
        AdoShowRequestsSupportedOutputFormats,
        typer.Option(
            "--output",
            "-o",
            help="The format in which to output the requests.",
        ),
    ] = AdoShowRequestsSupportedOutputFormats.TABLE.value,
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
    hide_fields: Annotated[
        list[str] | None,
        typer.Option(
            "--hide",
            show_default=False,
            help="""
            Hide columns (fields) from the output. Can be used multiple times.

            Different resource types might support different fields.
            """,
        ),
    ] = None,
    no_trunc: Annotated[
        bool,
        typer.Option(
            "--no-trunc",
            help="""
            Prevent truncation of table content. When enabled, columns will be sized to fit all content
            without truncation. Only applies to console output format.
            """,
        ),
    ] = False,
) -> None:
    """
    Show the timeseries of requests for an operation.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-requests
    for detailed documentation and examples.



    Examples:



    # Show the timeseries of requests for an operation

    ado show requests operation <operation-id>



    # Show the timeseries of requests for the latest operation

    ado show requests operation --use-latest



    # Show the timeseries of requests for an operation and hide the request id and metadata columns

    ado show requests operation <operation-id> --hide id --hide metadata
    """
    ado_configuration: AdoConfiguration = ctx.obj

    # Validate that output_file is only used with file-based formats
    if output_file and output_format == AdoShowRequestsSupportedOutputFormats.TABLE:
        console_print(
            f"{ERROR} --output-file cannot be used with --output console. "
            f"Use --output csv or --output json instead.",
            stderr=True,
        )
        raise typer.Exit(1)

    if not resource_id and not use_latest:
        console_print(
            f"{ERROR}You must specify either a resource id or the --use-latest flag",
            stderr=True,
        )
        raise typer.Exit(1)

    if use_latest:
        resource_id = get_effective_resource_id(
            explicit_resource_id=resource_id,
            resource_type=resource_type.value,
            project_context=ado_configuration.project_context,
        )

    parameters = AdoShowRequestsCommandParameters(
        ado_configuration=ado_configuration,
        hide_fields=hide_fields,
        no_trunc=no_trunc,
        output_file=output_file,
        output_format=output_format,
        resource_id=resource_id,
    )

    method_mapping = {
        AdoShowRequestsSupportedResourceTypes.OPERATION: show_operation_requests
    }

    method_mapping[resource_type](parameters=parameters)


def register_show_requests_command(app: typer.Typer) -> None:
    app.command(
        name="requests",
        no_args_is_help=True,
        options_metavar="[-o | --output <format>] [--output-file <path>] [--hide <column>]",
    )(show_requests_for_resources)
