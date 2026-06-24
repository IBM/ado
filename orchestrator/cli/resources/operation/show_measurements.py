# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowMeasurementsCommandParameters
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.dataframes import df_to_output
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)


def show_operation_measurements(
    parameters: AdoShowMeasurementsCommandParameters,
) -> None:
    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        try:
            space = DiscoverySpace.from_operation_id(
                operation_id=parameters.resource_id,
                project_context=parameters.ado_configuration.project_context,
                metadata_store=sql_store,
            )
        except (ResourceDoesNotExistError, NoRelatedResourcesError):
            status.stop()
            raise

        status.update("Fetching measurements")
        output_df = space.complete_measurement_request_with_results_timeseries(
            operation_id=parameters.resource_id,
            output_format=parameters.measurements_property_format.value,
            limit_to_properties=parameters.properties,
            aggregation_method=parameters.aggregation_method,
        )

    df_to_output(
        df=output_df,
        output_format=parameters.measurements_output_format.value,
        output_file=parameters.output_file,
        no_trunc=parameters.no_trunc,
    )
