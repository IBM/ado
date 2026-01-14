# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing

import pydantic
import typer
import yaml

from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import AdoGetSupportedOutputFormats
from orchestrator.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    ERROR,
    HINT,
    WARN,
    console_print,
    context_not_in_available_contexts_error_str,
    cyan,
)
from orchestrator.metastore.project import ProjectContext

if typing.TYPE_CHECKING:
    import pandas as pd


def get_context(
    parameters: AdoGetCommandParameters,
    simplify_output: bool = False,
) -> None:

    available_contexts = parameters.ado_configuration.available_contexts

    # AP 11/06/2025:
    # The only possible way for this should be when the user is
    # providing a context with -c and the ado context dir is empty
    if len(available_contexts) == 0:
        console_print(
            f"{WARN}There are no contexts available.\n"
            f"{HINT}You can create a context with {cyan('ado create context')}",
            stderr=True,
        )
        raise typer.Exit(1)

    if parameters.resource_id:
        if parameters.resource_id not in available_contexts:
            console_print(
                context_not_in_available_contexts_error_str(
                    requested_context=parameters.resource_id,
                    available_contexts=available_contexts,
                ),
                stderr=True,
            )
            raise typer.Exit(1)

        # We overwrite the available_contexts to handle both
        # single and multiple contexts with the same code
        available_contexts = [parameters.resource_id]

    # AP: we always want to dump default values for contexts
    parameters.exclude_default = False

    if parameters.output_format == AdoGetSupportedOutputFormats.DEFAULT:
        if simplify_output:
            _simple_contexts_formatting(contexts=available_contexts)
            return

        contexts_df = _prepare_context_dataframe(
            contexts=available_contexts,
            default_context=parameters.ado_configuration.active_context,
        )
        if contexts_df.empty:
            console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            return

        console_print(contexts_df)
        return

    from orchestrator.cli.utils.resources.formatters import (
        format_resource_for_ado_get_custom_format,
    )

    try:
        to_print = [
            ProjectContext.model_validate(
                yaml.safe_load(
                    parameters.ado_configuration.project_context_path_for_context(
                        ctx
                    ).read_text()
                )
            )
            for ctx in available_contexts
        ]
    except pydantic.ValidationError as e:
        console_print(f"{ERROR}One of the contexts was not valid:\n{e}", stderr=True)
        raise typer.Exit(1) from e

    # AP: it's more readable to write this than to
    # have an if/else to build to_print directly
    if parameters.resource_id:
        to_print = to_print[0]

    console_print(
        format_resource_for_ado_get_custom_format(
            to_print=to_print, parameters=parameters
        )
    )


def _simple_contexts_formatting(contexts: list[str]) -> None:
    for context in sorted(contexts):
        console_print(context)


def _prepare_context_dataframe(
    contexts: list[str], default_context: str | None
) -> "pd.DataFrame":
    import pandas as pd

    default_context_column = ["*" if ctx == default_context else "" for ctx in contexts]
    output_df = pd.DataFrame({"CONTEXT": contexts, "DEFAULT": default_context_column})

    # Sort contexts by name
    output_df = output_df.sort_values(by=["CONTEXT"], axis="rows")
    return output_df.reset_index(drop=True)
