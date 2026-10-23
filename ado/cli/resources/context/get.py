# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import typing

import pydantic
import typer
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]

from ado.cli.models.parameters import AdoGetCommandParameters
from ado.cli.models.types import AdoGetSupportedOutputFormats
from ado.cli.utils.output.prints import (
    ADO_NO_CONTEXT_AVAILABLE_ERROR,
    ERROR,
    HINT,
    console_print,
    context_not_in_available_contexts_error_str,
    cyan,
)
from ado.metastore.project import ProjectContext

if typing.TYPE_CHECKING:
    import pandas as pd


def get_context(
    parameters: AdoGetCommandParameters,
) -> None:

    # Never truncate the CONTEXT (name) column
    if not parameters.no_trunc:
        parameters.no_trunc = ["CONTEXT"]

    available_contexts = parameters.ado_configuration.available_contexts

    # AP 11/06/2025:
    # The only possible way for this should be when the user is
    # providing a context with -c and the ado context dir is empty
    if len(available_contexts) == 0:
        console_print(ADO_NO_CONTEXT_AVAILABLE_ERROR, stderr=True)
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

    # For NAME, TABLE and CSV formats, use DataFrame
    if parameters.output_format in {
        AdoGetSupportedOutputFormats.NAME,
        AdoGetSupportedOutputFormats.TABLE,
        AdoGetSupportedOutputFormats.CSV,
    }:
        contexts_df = _prepare_context_dataframe(
            contexts=available_contexts,
            active_context=parameters.ado_configuration.active_context,
            use_emoji_for_active=parameters.output_format
            != AdoGetSupportedOutputFormats.CSV,
        )

        from ado.cli.utils.resources.handlers import handle_ado_get

        handle_ado_get(parameters=parameters, dataframe=contexts_df)
        return

    # For structured formats (YAML, JSON, CONFIG), load full resources
    to_print = []
    try:
        for ctx in available_contexts:
            to_print.append(
                ProjectContext.model_validate(
                    yaml.load(
                        parameters.ado_configuration.project_context_path_for_context(
                            ctx
                        ).read_text(),
                        Loader=SafeLoader,
                    )
                )
            )
    except pydantic.ValidationError as e:
        console_print(
            f"{ERROR}Context {cyan(ctx)} was not valid:\n\n{e}\n\n"
            f"{HINT}You can manually update the context file at: '{parameters.ado_configuration.project_context_path_for_context(ctx)}'\n"
            f"\tAlternatively, delete the context with {cyan(f'ado delete context {ctx}')}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    # AP: it's more readable to write this than to
    # have an if/else to build to_print directly
    if parameters.resource_id:
        to_print = to_print[0]

    from ado.cli.utils.resources.handlers import handle_ado_get

    handle_ado_get(parameters=parameters, resources=to_print)


def _prepare_context_dataframe(
    contexts: list[str],
    active_context: str | None,
    use_emoji_for_active: bool = True,
) -> "pd.DataFrame":
    """Build the contexts DataFrame.

    Args:
        contexts: Names of the contexts to include.
        active_context: Name of the active context, if any.
        use_emoji_for_active: When True, the ACTIVE column marks the active
            context with an emoji code; when False, it holds booleans.

    Returns:
        A DataFrame with CONTEXT and ACTIVE columns, sorted by CONTEXT.
    """
    import pandas as pd

    output_df = pd.DataFrame(
        {"CONTEXT": contexts, "ACTIVE": [ctx == active_context for ctx in contexts]}
    )
    if use_emoji_for_active:
        output_df["ACTIVE"] = output_df["ACTIVE"].map(
            lambda is_active: ":white_check_mark:" if is_active else ""
        )

    # Sort contexts by name
    output_df = output_df.sort_values(by=["CONTEXT"], axis="rows")
    return output_df.reset_index(drop=True)
