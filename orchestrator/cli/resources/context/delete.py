# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pydantic
import typer
import yaml
from rich.prompt import Confirm

from orchestrator.cli.models.parameters import AdoDeleteCommandParameters
from orchestrator.cli.utils.output.prints import (
    ERROR,
    HINT,
    INFO,
    SUCCESS,
    WARN,
    console_print,
    context_not_in_available_contexts_error_str,
    cyan,
)
from orchestrator.metastore.project import ProjectContext


def delete_context(parameters: AdoDeleteCommandParameters) -> None:

    available_contexts = parameters.ado_configuration.available_contexts
    if parameters.resource_id not in available_contexts:
        console_print(
            context_not_in_available_contexts_error_str(
                requested_context=parameters.resource_id,
                available_contexts=available_contexts,
            ),
            stderr=True,
        )
        raise typer.Exit(1)

    configuration_file = parameters.ado_configuration.project_context_path_for_context(
        parameters.resource_id
    )

    try:
        context = ProjectContext.model_validate(
            yaml.safe_load(configuration_file.read_text())
        )
    except pydantic.ValidationError as e:
        console_print(
            f"{ERROR}The path provided is not a valid ado context file: {e}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    # AP: if the user hasn't done anything with the DB, the file will not exist
    if (
        context.metadataStore.scheme == "sqlite"
        and parameters.ado_configuration.local_db_path_for_context(
            parameters.resource_id
        ).exists()
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
            parameters.resource_id
        )
        if parameters.delete_local_db:
            console_print(f"{INFO}Deleting local db {local_db_path}\n", stderr=True)
            local_db_path.unlink()
        else:
            console_print(
                f"{INFO}Local db {local_db_path} will not be deleted.\n", stderr=True
            )

    configuration_file.unlink()
    console_print(SUCCESS, stderr=True)

    if parameters.resource_id == parameters.ado_configuration.active_context:
        parameters.ado_configuration.active_context = None
        parameters.ado_configuration.store()
        console_print(
            f"{WARN}{parameters.resource_id} was your default context.\n"
            f"{HINT}Set a different one with {cyan('ado context')} or {cyan('ado create context')}",
            stderr=True,
        )
