# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

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
        str,
        typer.Argument(
            ...,
            help="The id of the resource to show the request timeseries for.",
            show_default=False,
        ),
    ],
    output_format: Annotated[
        AdoShowRequestsSupportedOutputFormats,
        typer.Option(
            "-o", "--output-format", help="The format in which to output the requests."
        ),
    ] = AdoShowRequestsSupportedOutputFormats.CONSOLE.value,
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
):
    """
    Show the timeseries of requests for an operation.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-requests
    for detailed documentation and examples.



    Examples:



    # Show the timeseries of requests for an operation

    ado show requests operation <operation-id>



    # Show the timeseries of requests for an operation and hide the request id and metadata columns

    ado show requests operation <operation-id> --hide id --hide metadata
    """
    ado_configuration: AdoConfiguration = ctx.obj
    parameters = AdoShowRequestsCommandParameters(
        ado_configuration=ado_configuration,
        hide_fields=hide_fields,
        output_format=output_format,
        resource_id=resource_id,
    )

    method_mapping = {
        AdoShowRequestsSupportedResourceTypes.OPERATION: show_operation_requests
    }

    method_mapping[resource_type](parameters=parameters)


def register_show_requests_command(app: typer.Typer):
    app.command(
        name="requests",
        no_args_is_help=True,
        options_metavar="[-o | --output-format <format>] [--hide <column>]",
    )(show_requests_for_resources)
