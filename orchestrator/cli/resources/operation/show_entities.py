# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowEntitiesCommandParameters
from orchestrator.cli.utils.output.dataframes import df_to_output
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)


def show_operation_entities(parameters: AdoShowEntitiesCommandParameters) -> None:

    entities_type = "timeseries"

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        try:
            space = DiscoverySpace.from_operation_id(
                operation_id=parameters.resource_id,
                project_context=parameters.ado_configuration.project_context,
            )
        except (ResourceDoesNotExistError, NoRelatedResourcesError):
            status.stop()
            raise

        status.update("Fetching measurements")
        output_df = space.complete_measurement_request_with_results_timeseries(
            operation_id=parameters.resource_id,
            output_format=parameters.entities_property_format.value,
            limit_to_properties=parameters.properties,
        )

    file_name = f"{parameters.resource_id}_description_{entities_type}_{parameters.entities_property_format.value}.{parameters.entities_output_format.value}"
    df_to_output(
        df=output_df,
        output_format=parameters.entities_output_format.value,
        file_name=file_name,
    )
