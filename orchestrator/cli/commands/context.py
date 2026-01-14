# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import (
    AdoGetSupportedOutputFormats,
    AdoGetSupportedResourceTypes,
)
from orchestrator.cli.resources.context.activate import activate_context
from orchestrator.cli.resources.context.get import get_context
from orchestrator.cli.utils.output.prints import (
    ADO_NO_ACTIVE_CONTEXT_ERROR,
    console_print,
    magenta,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


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
    simple: Annotated[
        bool,
        typer.Option(
            "--simple", help="Display only context names.", show_default=False
        ),
    ] = False,
) -> None:
    """
    List available contexts.

    See https://ibm.github.io/ado/getting-started/ado/#ado-context
    for detailed documentation and examples.



    Examples:



    # View available contexts and active context

    ado contexts



    # List available contexts

    ado contexts --simple
    """
    ado_configuration: AdoConfiguration = ctx.obj

    parameters = AdoGetCommandParameters(
        ado_configuration=ado_configuration,
        exclude_default=True,
        exclude_fields=None,
        exclude_none=True,
        exclude_unset=True,
        field_selectors=[{}],
        from_sample_store=None,
        from_operation=None,
        from_space=None,
        matching_point=None,
        matching_space_id=None,
        matching_space=None,
        minimize_output=True,
        output_format=AdoGetSupportedOutputFormats.DEFAULT,
        resource_id=None,
        resource_type=AdoGetSupportedResourceTypes.CONTEXT,
        show_deprecated=False,
        show_details=False,
    )

    # NOTE: there will always be at least one context (local)
    get_context(
        parameters=parameters,
        simplify_output=simple,
    )

    if simple:
        return

    if ado_configuration.active_context is None:
        console_print(ADO_NO_ACTIVE_CONTEXT_ERROR, stderr=True)
        return

    console_print(
        f"\nThe active context is: {magenta(ado_configuration.active_context)}"
    )


def register_context_command(app: typer.Typer) -> None:
    app.command(
        name="context",
        options_metavar="",
    )(manage_contexts)


def register_contexts_command(app: typer.Typer) -> None:
    app.command(
        name="contexts",
        options_metavar="[--simple]",
    )(list_contexts)
