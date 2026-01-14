# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowResultsCommandParameters
from orchestrator.cli.utils.output.dataframes import df_to_output
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from orchestrator.schema.request import ReplayedMeasurement
from orchestrator.schema.result import InvalidMeasurementResult, ValidMeasurementResult


def show_operation_results(parameters: AdoShowResultsCommandParameters) -> None:

    class _COLUMN(enum.Enum):
        RESULT_UID = "Result UID"
        REQUEST_ID = "Request ID"
        REQUEST_INDEX = "Request Index"
        REQUEST_TYPE = "Request type"
        EXPERIMENT_ID = "Experiment ID"
        ENTITY_ID = "Entity ID"
        VALID = "Valid"
        NUMBER_PROPERTIES = "Number of Properties"
        INVALID_REASON = "Invalid Reason"
        METADATA = "Metadata"

    hidable_fields = {
        **dict.fromkeys(["result uid", "uid"], _COLUMN.RESULT_UID.value),
        **dict.fromkeys(["request id", "id"], _COLUMN.REQUEST_ID.value),
        **dict.fromkeys(["request index", "idx"], _COLUMN.REQUEST_INDEX.value),
        **dict.fromkeys(["type", "request type"], _COLUMN.REQUEST_TYPE.value),
        **dict.fromkeys(["experiment id"], _COLUMN.EXPERIMENT_ID.value),
        **dict.fromkeys(["entity", "entity id"], _COLUMN.ENTITY_ID.value),
        **dict.fromkeys(["valid"], _COLUMN.VALID.value),
        **dict.fromkeys(
            ["number of properties", "properties"], _COLUMN.NUMBER_PROPERTIES.value
        ),
        **dict.fromkeys(
            ["invalid", "invalid reason", "reason"], _COLUMN.INVALID_REASON.value
        ),
        **dict.fromkeys(["meta", "metadata"], _COLUMN.METADATA.value),
    }

    if parameters.hide_fields:
        for idx, field in enumerate(parameters.hide_fields):
            if field.lower() not in hidable_fields:
                console_print(
                    f"{ERROR}You can only hide the following fields (case insensitive): "
                    f"{list(hidable_fields.keys())}",
                    stderr=True,
                )
                raise typer.Exit(1)
            parameters.hide_fields[idx] = hidable_fields[field.lower()]

    file_name = f"measurement_results_for_operation_{parameters.resource_id}.{parameters.output_format.value}"
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        try:
            space = DiscoverySpace.from_operation_id(
                operation_id=parameters.resource_id,
                project_context=parameters.ado_configuration.project_context,
            )
        except (ResourceDoesNotExistError, NoRelatedResourcesError):
            status.stop()
            raise

        # We need to fetch the requests even when displaying the results
        # because we want to include the request ID and type.
        status.update("Fetching measurements")
        measurement_requests = space.measurement_requests_for_operation(
            operation_id=parameters.resource_id,
        )

    rows = []
    for request in measurement_requests:
        for result in request.measurements:

            base_row = {
                _COLUMN.RESULT_UID.value: result.uid,
                _COLUMN.REQUEST_ID.value: request.requestid,
                _COLUMN.REQUEST_INDEX.value: request.requestIndex,
                _COLUMN.REQUEST_TYPE.value: (
                    "replayed"
                    if isinstance(request, ReplayedMeasurement)
                    else "measured"
                ),
                _COLUMN.EXPERIMENT_ID.value: request.experimentReference,
                _COLUMN.ENTITY_ID.value: result.entityIdentifier,
                _COLUMN.VALID.value: isinstance(result, ValidMeasurementResult),
                _COLUMN.METADATA.value: result.metadata,
            }

            if isinstance(result, InvalidMeasurementResult):
                base_row[_COLUMN.INVALID_REASON.value] = result.reason
            elif isinstance(result, ValidMeasurementResult):
                base_row[_COLUMN.NUMBER_PROPERTIES.value] = len(
                    {m.property.identifier for m in result.measurements}
                )

            rows.append(base_row)

    import pandas as pd

    from orchestrator.utilities.pandas import reorder_dataframe_columns

    df = pd.DataFrame(rows)
    df = reorder_dataframe_columns(
        df=df, move_to_start=[], move_to_end=[_COLUMN.METADATA.value]
    )
    if parameters.hide_fields:
        df = df.drop(parameters.hide_fields, axis="columns")

    df_to_output(
        df=df, output_format=parameters.output_format.value, file_name=file_name
    )
