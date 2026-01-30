# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from rich.status import Status

from orchestrator.cli.models.parameters import AdoDescribeCommandParameters
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    console_print,
)
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.base import ResourceDoesNotExistError


def describe_data_container(parameters: AdoDescribeCommandParameters) -> None:

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        datacontainer_resource = sql.getResource(
            identifier=parameters.resource_id, kind=CoreResourceKinds.DATACONTAINER
        )
        if not datacontainer_resource:
            status.stop()
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id, kind=CoreResourceKinds.DATACONTAINER
            )

    console_print(datacontainer_resource)
