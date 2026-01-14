# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from typing import NoReturn

import typer

from orchestrator.cli.utils.output.prints import (
    console_print,
    could_not_delete_resource_from_database_error_str,
    no_related_resources_error_str,
    no_resource_with_id_in_db_error_str,
    unknown_experiment_error_str,
)
from orchestrator.metastore.base import (
    DeleteFromDatabaseError,
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.actuators.registry import UnknownExperimentError


def handle_resource_does_not_exist(
    error: ResourceDoesNotExistError, project_context: ProjectContext
) -> NoReturn:
    console_print(
        no_resource_with_id_in_db_error_str(
            resource_id=error.resource_id,
            kind=error.kind,
            context=project_context.project,
        ),
        stderr=True,
    )
    raise typer.Exit(1) from error


def handle_no_related_resource(
    error: NoRelatedResourcesError, project_context: ProjectContext
) -> NoReturn:
    console_print(
        no_related_resources_error_str(
            resource_id=error.resource_id,
            kind=error.kind,
            context=project_context.project,
        ),
        stderr=True,
    )
    raise typer.Exit(1) from error


def handle_unknown_experiment_error(error: UnknownExperimentError) -> NoReturn:
    console_print(unknown_experiment_error_str(error=error), stderr=True)
    raise typer.Exit(1) from error


def handle_resource_deletion_error(error: DeleteFromDatabaseError) -> NoReturn:
    console_print(
        could_not_delete_resource_from_database_error_str(
            error=error,
        ),
        stderr=True,
    )
    raise typer.Exit(1) from error
