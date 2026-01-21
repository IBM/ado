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


def show_operation_details(parameters: AdoShowDetailsCommandParameters) -> None:
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

            from orchestrator.schema.result import ValidMeasurementResult

            measurement_results_for_operation = space.measurement_results_for_operation(
                operation_id=parameters.resource_id
            )

            entities_with_all_successful_measurements = {
                result.entityIdentifier for result in measurement_results_for_operation
            }
            entities_with_at_least_one_successful_measurement = set()
            for measurement_result in measurement_results_for_operation:

                if isinstance(measurement_result, ValidMeasurementResult):
                    entities_with_at_least_one_successful_measurement.add(
                        measurement_result.entityIdentifier
                    )
                    continue

                entities_with_all_successful_measurements.discard(
                    measurement_result.entityIdentifier
                )

            table.add_row(
                "Total entities with no successful measurements",
                str(
                    total_entities_sampled
                    - len(entities_with_at_least_one_successful_measurement)
                ),
            )

            table.add_row(
                "Total entities with only partially successful measurements",
                str(
                    len(entities_with_at_least_one_successful_measurement)
                    - len(entities_with_all_successful_measurements)
                ),
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
