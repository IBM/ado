# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from ado.cli.models.parameters import AdoGetCommandParameters
from ado.cli.models.types import (
    AdoGetSupportedOutputFormats,
    AdoGetSupportedResourceTypes,
)
from ado.cli.resources.context.activate import activate_context
from ado.cli.resources.context.get import get_context
from ado.cli.utils.output.prints import (
    ADO_NO_ACTIVE_CONTEXT_ERROR,
    console_print,
)

if typing.TYPE_CHECKING:
    from ado.cli.core.config import AdoConfiguration


def manage_contexts(
    ctx: typer.Context,
    context_name: Annotated[
        str | None,
        typer.Argument(
            help="Optional name of the context to activate. Leave blank to print the current context.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """
    View or set the active context.

    See https://ibm.github.io/ado/getting-started/ado/#ado-context for
    detailed documentation and examples.

    Examples:

    # View the active context
    ado context

    # Set local as your active context
    ado context local
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if context_name:
        activate_context(context_name, ado_configuration)
        return

    if ado_configuration.active_context is None:
        console_print(ADO_NO_ACTIVE_CONTEXT_ERROR, stderr=True)
        raise typer.Exit(1)

    console_print(ado_configuration.active_context)


def list_contexts(
    ctx: typer.Context,
    output_format: Annotated[
        AdoGetSupportedOutputFormats,
        typer.Option(
            "--output",
            "-o",
            help="Output format. Use 'name' to display only context names.",
            show_default=False,
        ),
    ] = AdoGetSupportedOutputFormats.TABLE,
) -> None:
    """
    List available contexts.

    See https://ibm.github.io/ado/getting-started/ado/#ado-context
    for detailed documentation and examples.

    Examples:

    # View available contexts and active context
    ado contexts

    # List available context names only
    ado contexts -o name

    # Get contexts as YAML
    ado contexts -o yaml

    # Get contexts as JSON
    ado contexts -o json
    """
    ado_configuration: AdoConfiguration = ctx.obj

    parameters = AdoGetCommandParameters(
        ado_configuration=ado_configuration,
        exclude_default=True,
        exclude_fields=None,
        exclude_none=True,
        exclude_unset=True,
        field_selectors=[{}],
        matching_point=None,
        matching_space_id=None,
        matching_space=None,
        minimize_output=True,
        no_trunc=False,
        output_file=None,
        output_format=output_format,
        resource_id=None,
        resource_type=AdoGetSupportedResourceTypes.CONTEXT,
        show_deprecated=False,
        show_details=False,
        use_latest=False,
    )

    # NOTE: there will always be at least one context (local)
    get_context(parameters=parameters)

    # Warn user if no context is active only when using the TABLE format
    if (
        output_format == AdoGetSupportedOutputFormats.TABLE
        and ado_configuration.active_context is None
    ):
        console_print(ADO_NO_ACTIVE_CONTEXT_ERROR, stderr=True)


def register_context_command(app: typer.Typer) -> None:
    app.command(
        name="context",
        options_metavar="",
    )(manage_contexts)


def register_contexts_command(app: typer.Typer) -> None:
    app.command(
        name="contexts",
        options_metavar="[--output | -o <format>]",
    )(list_contexts)
