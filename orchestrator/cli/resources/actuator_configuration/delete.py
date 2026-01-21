# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoDeleteCommandParameters
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_DELETING_FROM_DB,
    ADO_SPINNER_QUERYING_DB,
    SUCCESS,
    cannot_delete_resource_due_to_children_resources,
    console_print,
)
from orchestrator.core import CoreResourceKinds
from orchestrator.metastore.base import (
    DeleteFromDatabaseError,
    ResourceDoesNotExistError,
)


def delete_actuator_configuration(parameters: AdoDeleteCommandParameters) -> None:

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as spinner:
        try:
            sql.getResource(
                identifier=parameters.resource_id,
                kind=CoreResourceKinds.ACTUATORCONFIGURATION,
                raise_error_if_no_resource=True,
            )
        except ResourceDoesNotExistError:
            spinner.stop()
            raise

        children_resources = sql.getRelatedObjectResourceIdentifiers(
            identifier=parameters.resource_id
        )

        if not children_resources.empty:
            spinner.stop()
            console_print(
                cannot_delete_resource_due_to_children_resources(
                    resource_kind=CoreResourceKinds.ACTUATORCONFIGURATION,
                    resource_id=parameters.resource_id,
                    children_resources=children_resources,
                ),
                stderr=True,
            )
            raise typer.Exit(1)

        spinner.update(ADO_SPINNER_DELETING_FROM_DB)
        try:
            sql.delete_actuator_configuration(parameters.resource_id)
        except DeleteFromDatabaseError:
            spinner.stop()
            raise

    console_print(SUCCESS, stderr=True)
