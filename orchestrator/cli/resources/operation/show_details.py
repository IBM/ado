# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from rich.status import Status

import orchestrator.cli.utils.resources.handlers
from orchestrator.cli.models.parameters import AdoShowDetailsCommandParameters
from orchestrator.cli.utils.generic.wrappers import (
    get_sql_store,
)
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_QUERYING_DB,
    console_print,
)
from orchestrator.core import OperationResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
)
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)


def show_operation_details(parameters: AdoShowDetailsCommandParameters):
    import rich.rule
    import rich.table

    table = rich.table.Table("", header_style=None, box=None)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )

    with Status(ADO_SPINNER_QUERYING_DB) as status:

        operation_conf = sql_store.getResource(
            identifier=parameters.resource_id, kind=CoreResourceKinds.OPERATION
        )
        if not operation_conf:
            status.stop()
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id, kind=CoreResourceKinds.OPERATION
            )

        try:
            space = DiscoverySpace.from_operation_id(
                operation_id=parameters.resource_id,
                project_context=parameters.ado_configuration.project_context,
            )
        except (ResourceDoesNotExistError, NoRelatedResourcesError):
            status.stop()
            raise

        status.update(ADO_SPINNER_GETTING_OUTPUT_READY)
        total_entities_sampled = operation_conf.metadata.get("entities_submitted", 0)
        table.add_row("Total entities sampled", str(total_entities_sampled))
        if (
            isinstance(operation_conf, OperationResource)
            and operation_conf.operationType == DiscoveryOperationEnum.SEARCH
        ):
            submitted_entities_field_name = "entities_submitted"
            submitted_entities = operation_conf.metadata.get(
                submitted_entities_field_name, 0
            )
            entities_with_no_successful_measurements = (
                submitted_entities - total_entities_sampled
            )
            table.add_row(
                "Total entities with no successful measurements",
                str(entities_with_no_successful_measurements),
            )
            table.add_row(
                "Total entities with partial successful measurements",
                str(total_entities_sampled - entities_with_no_successful_measurements),
            )

            entities_with_all_successful_measurements = set()
            for experiment in space.experiments_in_operation(
                operation_id=parameters.resource_id
            ):
                entity_identifiers_in_experiment = (
                    space.entity_identifiers_in_operation(
                        operation_id=parameters.resource_id
                    )
                )

                entities_with_all_successful_measurements = (
                    entities_with_all_successful_measurements.union(
                        entity_identifiers_in_experiment
                    )
                    if entities_with_all_successful_measurements
                    else entity_identifiers_in_experiment
                )

            table.add_row(
                "Total entities with all successful measurements",
                str(len(entities_with_all_successful_measurements)),
            )

    console_print(rich.rule.Rule(title="DETAILS"))
    console_print(table)
    orchestrator.cli.utils.resources.handlers.print_related_resources(
        resource_id=parameters.resource_id,
        resource_type=CoreResourceKinds.OPERATION,
        sql=sql_store,
    )
