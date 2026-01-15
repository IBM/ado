# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoDeleteCommandParameters
from orchestrator.cli.utils.generic.wrappers import (
    get_sql_store,
)
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_DELETING_FROM_DB,
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    HINT,
    SUCCESS,
    cannot_delete_resource_due_to_children_resources,
    console_print,
    cyan,
    magenta,
)
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.base import (
    DeleteFromDatabaseError,
    NotSupportedOnSQLiteError,
    ResourceDoesNotExistError,
    RunningOperationsPreventingDeletionError,
)


def delete_operation(parameters: AdoDeleteCommandParameters) -> None:

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as status:

        if not sql.containsResourceWithIdentifier(
            identifier=parameters.resource_id,
            kind=CoreResourceKinds.OPERATION,
        ):
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id, kind=CoreResourceKinds.OPERATION
            )

        children_resources = sql.getRelatedObjectResourceIdentifiers(
            identifier=parameters.resource_id
        )
        if not children_resources.empty:
            status.stop()
            console_print(
                cannot_delete_resource_due_to_children_resources(
                    resource_kind=CoreResourceKinds.OPERATION,
                    resource_id=parameters.resource_id,
                    children_resources=children_resources,
                ),
                stderr=True,
            )
            raise typer.Exit(1)

        status.update(ADO_SPINNER_DELETING_FROM_DB)
        try:
            sql.delete_operation(
                identifier=parameters.resource_id,
                ignore_running_operations=parameters.force,
            )
        except NotSupportedOnSQLiteError as e:
            status.stop()
            console_print(
                f"{ERROR}Checking for running operations using the same sample store as "
                f"operation {magenta(parameters.resource_id)} is not supported on local contexts.\n"
                f"{HINT}Make sure there are no such operations, and force the deletion by adding the "
                f"{cyan('--force')} flag.",
                stderr=True,
            )
            raise typer.Exit(1) from e
        except RunningOperationsPreventingDeletionError as e:
            status.stop()
            console_print(
                f"{ERROR}Operation {magenta(e.operation_id)} cannot be deleted "
                f"because the following operations have started and have not completed: "
                f"{e.running_operations}\n"
                f"{HINT}You can force the deletion by adding the {cyan('--force')} flag.",
                stderr=True,
            )
            raise typer.Exit(1) from e
        except DeleteFromDatabaseError:
            status.stop()
            raise

    console_print(SUCCESS)
