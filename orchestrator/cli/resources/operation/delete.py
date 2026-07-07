# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from rich.status import Status

from orchestrator.cli.models.parameters import AdoDeleteCommandParameters
from orchestrator.cli.utils.generic.wrappers import (
    get_sql_store,
)
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_DELETING_FROM_DB,
    ADO_SPINNER_QUERYING_DB,
)
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.base import (
    ResourceDoesNotExistError,
    ResourceHasChildrenError,
)


def delete_operation(parameters: AdoDeleteCommandParameters) -> None:
    """Delete a single operation.

    Args:
        parameters: Delete command parameters containing the operation id and options.

    Raises:
        ResourceDoesNotExistError: If the operation does not exist.
        ResourceHasChildrenError: If the operation has dependent resources.
        NotSupportedOnSQLiteError: If running-operations check is not supported on SQLite.
        RunningOperationsPreventingDeletionError: If running operations are using the same
            sample store.
        DeleteFromDatabaseError: If a database error occurs during deletion.
    """
    resource_id = parameters.resource_ids[0]

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if not sql.containsResourceWithIdentifier(
            identifier=resource_id,
            kind=CoreResourceKinds.OPERATION,
        ):
            raise ResourceDoesNotExistError(
                resource_id=resource_id, kind=CoreResourceKinds.OPERATION
            )

        children_resources = sql.getRelatedObjectResourceIdentifiers(
            identifier=resource_id
        )
        if not children_resources.empty:
            status.stop()
            raise ResourceHasChildrenError(
                resource_id=resource_id,
                kind=CoreResourceKinds.OPERATION,
                children_resources=children_resources,
            )

        status.update(ADO_SPINNER_DELETING_FROM_DB)
        try:
            sql.delete_operation(
                identifier=resource_id,
                ignore_running_operations=parameters.force,
            )
        except Exception:
            status.stop()
            raise
