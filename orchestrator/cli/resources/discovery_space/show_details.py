# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import rich.rule
from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowDetailsCommandParameters
from orchestrator.cli.models.space import SpaceDetails
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_INITIALIZING_DISCOVERY_SPACE,
    ADO_SPINNER_QUERYING_DB,
    console_print,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.base import ResourceDoesNotExistError


def show_discovery_space_details(parameters: AdoShowDetailsCommandParameters) -> None:
    import orchestrator.cli.utils.resources.handlers

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        space_resource = sql.getResource(
            identifier=parameters.resource_id, kind=CoreResourceKinds.DISCOVERYSPACE
        )
        if not space_resource:
            status.stop()
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id,
                kind=CoreResourceKinds.DISCOVERYSPACE,
            )

        status.update(ADO_SPINNER_INITIALIZING_DISCOVERY_SPACE)
        space = DiscoverySpace.from_configuration(
            conf=space_resource.config,
            project_context=parameters.ado_configuration.project_context,
            identifier=parameters.resource_id,
        )

    console_print(rich.rule.Rule(title="DETAILS"))
    console_print(SpaceDetails.from_space(space=space).to_markdown())
    orchestrator.cli.utils.resources.handlers.print_related_resources(
        resource_id=parameters.resource_id,
        resource_type=CoreResourceKinds.DISCOVERYSPACE,
        sql=sql,
    )
