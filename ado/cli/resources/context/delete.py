# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import yaml
from rich.prompt import Confirm

from ado.cli.models.parameters import AdoDeleteCommandParameters
from ado.cli.utils.output.prints import HINT, INFO, WARN, console_print, cyan
from ado.metastore.base import ContextDoesNotExistError
from ado.utilities.dictionaries import get_nested_value


def delete_context(parameters: AdoDeleteCommandParameters) -> None:
    """Delete a single context.

    Args:
        parameters: Delete command parameters containing the context id and options.

    Raises:
        ContextDoesNotExistError: If the requested context does not exist.
    """
    resource_id = parameters.resource_ids[0]

    available_contexts = parameters.ado_configuration.available_contexts
    if resource_id not in available_contexts:
        raise ContextDoesNotExistError(
            resource_id=resource_id,
            available_contexts=available_contexts,
        )

    configuration_file = parameters.ado_configuration.project_context_path_for_context(
        resource_id
    )
    context_dict = yaml.safe_load(configuration_file.read_text())

    # AP: the db might not exist if the user has never used the local context
    if (
        get_nested_value(context_dict, "metadataStore.scheme") == "sqlite"
        and parameters.ado_configuration.local_db_path_for_context(resource_id).exists()
    ):
        if parameters.delete_local_db is None:
            parameters.delete_local_db = Confirm.ask(
                f"{WARN}You are trying to delete a local context. Do you also wish to delete the local database?",
            )
            if parameters.delete_local_db:
                parameters.delete_local_db = Confirm.ask(
                    f"{WARN}Are you sure? This action cannot be undone.",
                )

        local_db_path = parameters.ado_configuration.local_db_path_for_context(
            resource_id
        )
        if parameters.delete_local_db:
            console_print(f"{INFO}Deleting local db {local_db_path}\n", stderr=True)
            local_db_path.unlink()
        else:
            console_print(
                f"{INFO}Local db {local_db_path} will not be deleted.\n", stderr=True
            )

    configuration_file.unlink()

    if resource_id == parameters.ado_configuration.active_context:
        parameters.ado_configuration.active_context = None
        parameters.ado_configuration.store()
        console_print(
            f"{WARN}{resource_id} was your default context.\n"
            f"{HINT}Set a different one with {cyan('ado context')} or {cyan('ado create context')}",
            stderr=True,
        )
