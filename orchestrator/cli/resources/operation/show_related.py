# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.cli.models.parameters import (
    AdoShowRelatedCommandParameters,
)
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.resources.handlers import (
    print_related_resources,
)
from orchestrator.core.resources import CoreResourceKinds


def show_resources_related_to_operation(
    parameters: AdoShowRelatedCommandParameters,
) -> None:
    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    print_related_resources(
        resource_id=parameters.resource_id,
        resource_type=CoreResourceKinds.OPERATION,
        sql=sql_store,
        hide_banner=True,
    )
