# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.cli.utils.output.prints import (
    ERROR,
    SUCCESS,
    WARN,
    console_print,
    context_not_in_available_contexts_error_str,
    green,
    magenta,
)


def activate_context(context_name: str, ado_configuration: AdoConfiguration) -> None:

    available_contexts = ado_configuration.available_contexts
    if context_name not in available_contexts:
        console_print(
            context_not_in_available_contexts_error_str(
                requested_context=context_name, available_contexts=available_contexts
            ),
            stderr=True,
        )
        raise typer.Exit(1)

    if ado_configuration.active_context == context_name:
        console_print(f"Context {context_name} is already active.", stderr=True)
        return

    # Do not allow switching to an invalid context
    try:
        ado_configuration.project_context_model_for_context(context_name)
    except ValueError as e:
        context_path = ado_configuration.project_context_path_for_context(context_name)
        console_print(
            f"{ERROR}Context {magenta(context_name)} is not valid:\n\n{e}\n\n"
            f"{WARN}You must fix the context manually: {green(context_path)}"
        )
        raise typer.Exit(1) from e

    ado_configuration.active_context = context_name
    ado_configuration.store()
    console_print(f"{SUCCESS}Now using context {context_name}", stderr=True)
