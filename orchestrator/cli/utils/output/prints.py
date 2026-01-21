# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing
from typing import Any

from rich.console import Console

from orchestrator.cli.utils.resources.mappings import (
    resource_kinds_to_human,
)
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.base import DeleteFromDatabaseError
from orchestrator.modules.actuators.registry import UnknownExperimentError

if typing.TYPE_CHECKING:
    import pandas as pd

ERROR = "[b red]ERROR[/b red]:\t"
WARN = "[b yellow]WARN[/b yellow]:\t"
HINT = "[b magenta]HINT[/b magenta]:\t"
INFO = "[b turquoise2]INFO[/b turquoise2]:\t"
SUCCESS = "[b green]Success![/b green] "

# ado create {} --dry-run
ADO_CREATE_DRY_RUN_CONFIG_VALID = (
    "[b green]The configuration passed is valid![/b green]"
)

# ado get related
ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE = (
    "The config output format can only be used when a resource identifier is provided."
)
ADO_INFO_EMPTY_DATAFRAME = "[b turquoise2]INFO[/b turquoise2]:\tNothing was returned"

ADO_NO_ACTIVE_CONTEXT_ERROR = (
    "[b yellow]WARN[/b yellow]:\tThere is no active context.\n"
    "[b magenta]HINT[/b magenta]:\tYou can create one with [b cyan]ado create context[/b cyan] "
    "or activate one with [b cyan]ado context[/b cyan]"
)

# Spinners
ADO_SPINNER_CONNECTING_TO_DB = "Connecting to the database"
ADO_SPINNER_QUERYING_DB = "Querying the database"
ADO_SPINNER_GETTING_OUTPUT_READY = "Getting the output ready"
ADO_SPINNER_SAVING_TO_DB = "Saving in the database"
ADO_SPINNER_DELETING_FROM_DB = "Deleting from database"
ADO_SPINNER_INITIALIZING_DISCOVERY_SPACE = "Initializing Discovery Space"
ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY = "Initializing Actuator Registry"

# Consoles
stdout_console = Console()
stderr_console = Console(stderr=True)


def set_pandas_display_options() -> None:
    import pandas as pd

    # Ensure we are extremely unlikely to truncate output in pandas
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1_000_000)
    pd.set_option("expand_frame_repr", False)


def console_print(
    *args: Any,  # noqa: ANN401
    stderr: bool = False,
    use_markup: bool = True,
    has_pandas_content: bool = False,
) -> None:
    import sys

    if has_pandas_content:
        import pandas as pd

        set_pandas_display_options()

        # AP: 08-12-2025
        # rich has issues when printing large dataframes using overflow="ignore"
        # ref: https://github.com/IBM/ado/issues/296
        if all(isinstance(item, pd.DataFrame) for item in args):
            print(*args, file=sys.stderr if stderr else sys.stdout)
            return

    if stderr:
        stderr_console.print(*args, overflow="ignore", crop=False, markup=use_markup)
    else:
        stdout_console.print(*args, overflow="ignore", crop=False, markup=use_markup)


# Template prints
def no_resource_with_id_in_db_error_str(
    resource_id: str, kind: CoreResourceKinds, context: str
) -> str:
    kind_specifier = f" and kind {magenta(kind.value)}"
    return (
        f"{ERROR}The database does not contain a resource with id {magenta(resource_id)}{kind_specifier}.\n"
        f"{HINT}Your active context is {cyan(context)} - are you sure it's the correct one?\n"
        f"\tYou can change it with {cyan('ado context')}"
    )


def no_related_resources_error_str(
    resource_id: str,
    kind: CoreResourceKinds,
    context: str,
) -> str:
    return (
        f"{ERROR}Resource with id {magenta(resource_id)} does not have related resources of kind {kind.value}.\n"
        f"{HINT}Your active context is {cyan(context)} - are you sure it's the correct one?\n"
        f"\tYou can change it with {cyan('ado context')}"
    )


def context_not_in_available_contexts_error_str(
    requested_context: str, available_contexts: list[str]
) -> str:
    return (
        f"{ERROR}{requested_context} is not in the available contexts.\n"
        f"{HINT}The available contexts are {sorted(available_contexts)}"
    )


def unknown_experiment_error_str(error: UnknownExperimentError) -> str:
    return (
        f"{ERROR}The following experiment was not found: {error}\n"
        f"{HINT}Check available experiments with {cyan('ado get actuators --details')}"
    )


def cannot_delete_resource_due_to_children_resources(
    resource_kind: CoreResourceKinds,
    resource_id: str,
    children_resources: "pd.DataFrame",
) -> str:
    return (
        f"{ERROR}Cannot delete {resource_kind.value} {magenta(resource_id)} "
        f"as it has children resources:\n\n{children_resources}\n\n"
        f"{HINT}You must delete each of them them first."
    )


def could_not_delete_resource_from_database_error_str(
    error: DeleteFromDatabaseError,
) -> str:

    transaction_rollback_message = (
        f"{INFO}The transaction has been rolled back"
        if error.rollback_occurred
        else f"{WARN}The transaction could not be rolled back"
    )
    optional_message = f"{WARN}{error.message}\n" if error.message else ""

    error_to_be_displayed = error.__cause__
    if not error.__cause__:
        import warnings

        warnings.warn(
            f"{error.__class__.__name__} should be raised from another error",
            stacklevel=2,
        )
        error_to_be_displayed = error

    return (
        f"{ERROR}An exception occurred when attempting to delete "
        f"{error.resource_kind.value} {magenta(error.resource_id)}:\n\n"
        f"{error_to_be_displayed}\n\n{optional_message}{transaction_rollback_message}"
    )


def latest_identifier_for_resource_not_found(
    resource_kind: CoreResourceKinds, hide_resource_in_flag: bool = False
) -> str:
    resource_human_readable_name = resource_kinds_to_human[resource_kind]
    flag = (
        cyan("--use-latest")
        if hide_resource_in_flag
        else cyan(f"--use-latest {resource_kind.value}")
    )

    return (
        f"{ERROR}Unable to find the identifier of the latest {resource_human_readable_name}.\n\t"
        f"This means that the {flag} flag cannot be used.\n"
        f"{HINT}Try creating a new {resource_human_readable_name} with {cyan(f'ado create {resource_kind.value}')}"
    )


def value_in_configuration_replaced_with_latest_identifier_for_resource(
    reused_resource_kind: CoreResourceKinds,
    target_resource_kind: CoreResourceKinds,
    replacement_identifier: str,
) -> str:
    reused_resource_human_readable_name = resource_kinds_to_human[reused_resource_kind]
    target_resource_human_readable_name = resource_kinds_to_human[target_resource_kind]
    return (
        f"{INFO}The latest {magenta(reused_resource_human_readable_name)} was requested to be reused.\n"
        f"\t{reused_resource_human_readable_name.title()}s referenced in the {target_resource_human_readable_name} "
        f"definition will be ignored and replaced with {cyan(replacement_identifier)}."
    )


def using_latest_identifier_for_resource(
    resource_kind: CoreResourceKinds, resource_identifier: str
) -> str:
    latest_resource_human_readable_name = resource_kinds_to_human[resource_kind]
    return f"{INFO}Using {latest_resource_human_readable_name} {magenta(resource_identifier)}."


# Styles
def bold(input: str) -> str:
    return f"[b]{input}[/b]"


# Colours
def magenta(input: str) -> str:
    return f"[b magenta]{input}[/b magenta]"


def cyan(input: str) -> str:
    return f"[b cyan]{input}[/b cyan]"


def green(input: str) -> str:
    return f"[b green]{input}[/b green]"
