# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.cli.models.parameters import (
    AdoShowRelatedCommandParameters,
)
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.resources.handlers import (
    print_related_resources,
)
from ado.core.resources import CoreResourceKinds


def show_resources_related_to_data_container(
    parameters: AdoShowRelatedCommandParameters,
) -> None:
    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    print_related_resources(
        resource_id=parameters.resource_id,
        resource_type=CoreResourceKinds.DATACONTAINER,
        sql=sql_store,
        hide_banner=True,
        max_hops=parameters.max_hops,
    )
