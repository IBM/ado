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


def delete_discovery_space(parameters: AdoDeleteCommandParameters) -> None:
    """Delete a single discovery space.

    Args:
        parameters: Delete command parameters containing the discovery space id.

    Raises:
        ResourceDoesNotExistError: If the discovery space does not exist.
        ResourceHasChildrenError: If the discovery space has dependent resources.
        DeleteFromDatabaseError: If a database error occurs during deletion.
    """
    resource_id = parameters.resource_ids[0]

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if not sql.containsResourceWithIdentifier(
            identifier=resource_id,
            kind=CoreResourceKinds.DISCOVERYSPACE,
        ):
            status.stop()
            raise ResourceDoesNotExistError(
                resource_id=resource_id,
                kind=CoreResourceKinds.DISCOVERYSPACE,
            )

        children_resources = sql.getRelatedObjectResourceIdentifiers(
            identifier=resource_id
        )
        if not children_resources.empty:
            status.stop()
            raise ResourceHasChildrenError(
                resource_id=resource_id,
                kind=CoreResourceKinds.DISCOVERYSPACE,
                children_resources=children_resources,
            )

        status.update(ADO_SPINNER_DELETING_FROM_DB)
        try:
            sql.delete_discovery_space(identifier=resource_id)
        except Exception:
            status.stop()
            raise
