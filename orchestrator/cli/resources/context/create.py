# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

import pydantic
import typer
import yaml

from orchestrator.cli.models.parameters import AdoCreateCommandParameters
from orchestrator.cli.utils.output.prints import (
    ADO_CREATE_DRY_RUN_CONFIG_VALID,
    ERROR,
    INFO,
    SUCCESS,
    console_print,
    cyan,
)
from orchestrator.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.location import (
    SQLiteStoreConfiguration,
)


def create_context(parameters: AdoCreateCommandParameters) -> str | None:

    try:
        context_configuration = ProjectContext.model_validate(
            yaml.safe_load(parameters.resource_configuration_file.read_text())
        )
    except pydantic.ValidationError as e:
        console_print(f"{ERROR}The context provided was not valid:\n{e}", stderr=True)
        raise typer.Exit(1) from e

    if parameters.override_values:
        context_configuration = override_values_in_pydantic_model(
            model=context_configuration, override_values=parameters.override_values
        )

    if parameters.dry_run:
        console_print(ADO_CREATE_DRY_RUN_CONFIG_VALID, stderr=True)
        return None

    destination = parameters.ado_configuration.project_context_path_for_context(
        context_configuration.project
    )

    pathlib.Path.mkdir(destination.parent, parents=True, exist_ok=True)
    if isinstance(context_configuration.metadataStore, SQLiteStoreConfiguration):
        try:
            pathlib.Path.mkdir(
                pathlib.Path(context_configuration.metadataStore.path).parent,
                parents=True,
                exist_ok=True,
            )
        except OSError as e:
            console_print(f"{ERROR}Unable to create DB directory: {e}", stderr=True)
            raise typer.Exit(1) from e

        if pathlib.Path(context_configuration.metadataStore.path).exists():
            console_print(
                f"{INFO}Found a pre-existing database at {context_configuration.metadataStore.path}",
                stderr=True,
            )

    destination.write_text(context_configuration.model_dump_json())
    console_print(SUCCESS, stderr=True)

    if parameters.ado_configuration.active_context is None:
        parameters.ado_configuration.active_context = context_configuration.project
        parameters.ado_configuration.store()
        console_print(
            f"{INFO}{context_configuration.project} is now your default context",
            stderr=True,
        )
    else:
        console_print(
            f"{INFO}To set it as your default context, run:\n"
            f"\t{cyan('ado context ' + context_configuration.project)}",
            stderr=True,
        )

    return context_configuration.project
