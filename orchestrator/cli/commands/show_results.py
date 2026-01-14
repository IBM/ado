# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoShowResultsCommandParameters
from orchestrator.cli.models.types import (
    AdoShowResultsSupportedOutputFormats,
    AdoShowResultsSupportedResourceTypes,
)
from orchestrator.cli.resources.operation.show_results import show_operation_results
from orchestrator.cli.utils.generic.common import get_effective_resource_id
from orchestrator.cli.utils.output.prints import ERROR, console_print

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


def show_results_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowResultsSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show the result timeseries for.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoShowResultsSupportedResourceTypes),
        ),
    ],
    resource_id: Annotated[
        str | None,
        typer.Argument(
            help="The id of the resource to show the result timeseries for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Show the timeseries of results for the latest identifier of the selected resource type. "
            "Ignored if a resource identifier is also specified.",
            show_default=False,
        ),
    ] = False,
    output_format: Annotated[
        AdoShowResultsSupportedOutputFormats,
        typer.Option(
            "-o", "--output-format", help="The format in which to output the results."
        ),
    ] = AdoShowResultsSupportedOutputFormats.CONSOLE.value,
    hide_fields: Annotated[
        list[str] | None,
        typer.Option(
            "--hide",
            help="Fields to hide from the output. "
            "Different resource types might support different fields.",
        ),
    ] = None,
) -> None:
    """
    Show the timeseries of results for an operation.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-results
    for detailed documentation and examples.



    Examples:



    # Show the timeseries of results for an operation

    ado show results operation <operation-id>



    # Show the timeseries of results for the latest operation

    ado show results operation --use-latest



    # Show the timeseries of results for an operation and hide the result uid

    ado show results operation <operation-id> --hide uid
    """
    ado_configuration: AdoConfiguration = ctx.obj

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
            ado_configuration=ado_configuration,
        )

    parameters = AdoShowResultsCommandParameters(
        ado_configuration=ado_configuration,
        hide_fields=hide_fields,
        output_format=output_format,
        resource_id=resource_id,
    )

    method_mapping = {
        AdoShowResultsSupportedResourceTypes.OPERATION: show_operation_results
    }

    method_mapping[resource_type](parameters=parameters)


def register_show_results_command(app: typer.Typer) -> None:
    app.command(
        name="results",
        no_args_is_help=True,
        options_metavar="[-o | --output-format <format>] [--hide <column>]",
    )(show_results_for_resources)
