# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.cli.utils.output.prints import (
    SUCCESS,
    console_print,
    context_not_in_available_contexts_error_str,
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

    ado_configuration.active_context = context_name
    ado_configuration.store()
    console_print(f"{SUCCESS}Now using context {context_name}", stderr=True)
