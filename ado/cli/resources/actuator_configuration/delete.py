# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from rich.status import Status

from ado.cli.models.parameters import AdoDeleteCommandParameters
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_SPINNER_DELETING_FROM_DB,
    ADO_SPINNER_QUERYING_DB,
)
from ado.core import CoreResourceKinds
from ado.metastore.base import (
    ResourceDoesNotExistError,
    ResourceHasChildrenError,
)


def delete_actuator_configuration(parameters: AdoDeleteCommandParameters) -> None:
    """Delete a single actuator configuration.

    Args:
        parameters: Delete command parameters containing the actuator configuration id.

    Raises:
        ResourceDoesNotExistError: If the actuator configuration does not exist.
        ResourceHasChildrenError: If the actuator configuration has dependent resources.
        DeleteFromDatabaseError: If a database error occurs during deletion.
    """
    resource_id = parameters.resource_ids[0]

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as spinner:
        if not sql.containsResourceWithIdentifier(
            identifier=resource_id,
            kind=CoreResourceKinds.ACTUATORCONFIGURATION,
        ):
            spinner.stop()
            raise ResourceDoesNotExistError(
                resource_id=resource_id,
                kind=CoreResourceKinds.ACTUATORCONFIGURATION,
            )

        children_resources = sql.getRelatedObjectResourceIdentifiers(
            identifier=resource_id
        )

        if not children_resources.empty:
            spinner.stop()
            raise ResourceHasChildrenError(
                resource_id=resource_id,
                kind=CoreResourceKinds.ACTUATORCONFIGURATION,
                children_resources=children_resources,
            )

        spinner.update(ADO_SPINNER_DELETING_FROM_DB)
        try:
            sql.delete_actuator_configuration(resource_id)
        except Exception:
            spinner.stop()
            raise
