# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowTraceCommandParameters
from orchestrator.cli.models.types import AdoShowTraceSupportedOutputFormats
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.dataframes import df_to_output
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from orchestrator.core.samplestore.base import SampleStore
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from orchestrator.schema.request import ReplayedMeasurement
from orchestrator.schema.result import InvalidMeasurementResult, ValidMeasurementResult
from orchestrator.utilities.output import pydantic_model_as_yaml


# Column definitions for request-level view
class _REQUEST_COLUMN(enum.Enum):
    REQUEST_ID = "Request ID"
    REQUEST_INDEX = "Request Index"
    REQUEST_TYPE = "Request type"
    TIMESTAMP = "Timestamp"
    EXPERIMENT_ID = "Experiment ID"
    ENTITY_IDS = "Entity IDs"
    STATUS = "Status"
    MEASUREMENTS = "Measurements"
    VALID_MEASUREMENTS = "Valid Measurements"
    INVALID_MEASUREMENTS = "Invalid Measurements"
    METADATA = "Metadata"


# Column definitions for result-level view
class _RESULT_COLUMN(enum.Enum):
    REQUEST_ID = "Request ID"
    REQUEST_INDEX = "Request Index"
    REQUEST_TYPE = "Request type"
    TIMESTAMP = "Timestamp"
    EXPERIMENT_ID = "Experiment ID"
    RESULT_INDEX = "Result Index"
    RESULT_UID = "Result UID"
    ENTITY_ID = "Entity ID"
    VALID = "Valid"
    NUMBER_OF_PROPERTIES = "Number of Properties"
    INVALID_REASON = "Invalid Reason"
    REQUEST_METADATA = "Request Metadata"
    RESULT_METADATA = "Result Metadata"


# Hidable fields mapping for request view
_REQUEST_HIDABLE_FIELDS = {
    **dict.fromkeys(["request id", "id"], _REQUEST_COLUMN.REQUEST_ID.value),
    **dict.fromkeys(["request index"], _REQUEST_COLUMN.REQUEST_INDEX.value),
    **dict.fromkeys(["type", "request type"], _REQUEST_COLUMN.REQUEST_TYPE.value),
    **dict.fromkeys(["timestamp", "time"], _REQUEST_COLUMN.TIMESTAMP.value),
    **dict.fromkeys(
        ["experiment id", "experiment"], _REQUEST_COLUMN.EXPERIMENT_ID.value
    ),
    **dict.fromkeys(
        ["entity", "entities", "entity ids"], _REQUEST_COLUMN.ENTITY_IDS.value
    ),
    **dict.fromkeys(["status"], _REQUEST_COLUMN.STATUS.value),
    **dict.fromkeys(["measurements"], _REQUEST_COLUMN.MEASUREMENTS.value),
    **dict.fromkeys(
        ["valid", "valid measurements"], _REQUEST_COLUMN.VALID_MEASUREMENTS.value
    ),
    **dict.fromkeys(
        ["invalid", "invalid measurements"],
        _REQUEST_COLUMN.INVALID_MEASUREMENTS.value,
    ),
    **dict.fromkeys(["meta", "metadata"], _REQUEST_COLUMN.METADATA.value),
}

# Hidable fields mapping for result view
_RESULT_HIDABLE_FIELDS = {
    **dict.fromkeys(["request id", "id"], _RESULT_COLUMN.REQUEST_ID.value),
    **dict.fromkeys(["request index"], _RESULT_COLUMN.REQUEST_INDEX.value),
    **dict.fromkeys(["type", "request type"], _RESULT_COLUMN.REQUEST_TYPE.value),
    **dict.fromkeys(["timestamp", "time"], _RESULT_COLUMN.TIMESTAMP.value),
    **dict.fromkeys(
        ["experiment id", "experiment"], _RESULT_COLUMN.EXPERIMENT_ID.value
    ),
    **dict.fromkeys(["result index"], _RESULT_COLUMN.RESULT_INDEX.value),
    **dict.fromkeys(["result uid", "uid"], _RESULT_COLUMN.RESULT_UID.value),
    **dict.fromkeys(["entity", "entity id"], _RESULT_COLUMN.ENTITY_ID.value),
    **dict.fromkeys(["valid"], _RESULT_COLUMN.VALID.value),
    **dict.fromkeys(
        ["number of properties", "properties"],
        _RESULT_COLUMN.NUMBER_OF_PROPERTIES.value,
    ),
    **dict.fromkeys(
        ["invalid reason", "reason", "invalid"],
        _RESULT_COLUMN.INVALID_REASON.value,
    ),
    **dict.fromkeys(
        ["request metadata", "request meta"],
        _RESULT_COLUMN.REQUEST_METADATA.value,
    ),
    **dict.fromkeys(
        ["result metadata", "result meta", "meta", "metadata"],
        _RESULT_COLUMN.RESULT_METADATA.value,
    ),
}


def _build_request_level_rows(
    measurement_requests: list,
) -> list[dict]:
    """Build rows for request-level view."""
    return [
        {
            _REQUEST_COLUMN.REQUEST_ID.value: request.requestid,
            _REQUEST_COLUMN.REQUEST_INDEX.value: request.requestIndex,
            _REQUEST_COLUMN.REQUEST_TYPE.value: (
                "replayed" if isinstance(request, ReplayedMeasurement) else "measured"
            ),
            _REQUEST_COLUMN.TIMESTAMP.value: request.timestamp,
            _REQUEST_COLUMN.EXPERIMENT_ID.value: request.experimentReference,
            _REQUEST_COLUMN.ENTITY_IDS.value: [
                entity.identifier for entity in request.entities
            ],
            _REQUEST_COLUMN.STATUS.value: request.status.value,
            _REQUEST_COLUMN.MEASUREMENTS.value: len(request.measurements or []),
            _REQUEST_COLUMN.VALID_MEASUREMENTS.value: len(
                [
                    r
                    for r in (request.measurements or [])
                    if isinstance(r, ValidMeasurementResult)
                ]
            ),
            _REQUEST_COLUMN.INVALID_MEASUREMENTS.value: len(
                [
                    r
                    for r in (request.measurements or [])
                    if isinstance(r, InvalidMeasurementResult)
                ]
            ),
            _REQUEST_COLUMN.METADATA.value: request.metadata,
        }
        for request in measurement_requests
    ]


def _build_result_level_rows(
    measurement_requests: list,
) -> list[dict]:
    """Build rows for result-level view with unrolled entities."""
    rows = []
    for request in measurement_requests:
        for result_idx, result in enumerate(request.measurements or []):
            row = {
                _RESULT_COLUMN.REQUEST_ID.value: request.requestid,
                _RESULT_COLUMN.REQUEST_INDEX.value: request.requestIndex,
                _RESULT_COLUMN.REQUEST_TYPE.value: (
                    "replayed"
                    if isinstance(request, ReplayedMeasurement)
                    else "measured"
                ),
                _RESULT_COLUMN.TIMESTAMP.value: request.timestamp,
                _RESULT_COLUMN.EXPERIMENT_ID.value: request.experimentReference,
                _RESULT_COLUMN.RESULT_INDEX.value: result_idx,
                _RESULT_COLUMN.RESULT_UID.value: result.uid,
                _RESULT_COLUMN.ENTITY_ID.value: result.entityIdentifier,
                _RESULT_COLUMN.VALID.value: isinstance(result, ValidMeasurementResult),
                _RESULT_COLUMN.REQUEST_METADATA.value: request.metadata,
                _RESULT_COLUMN.RESULT_METADATA.value: result.metadata,
            }

            if isinstance(result, ValidMeasurementResult):
                row[_RESULT_COLUMN.NUMBER_OF_PROPERTIES.value] = len(
                    {m.property.identifier for m in result.measurements}
                )
            elif isinstance(result, InvalidMeasurementResult):
                row[_RESULT_COLUMN.INVALID_REASON.value] = result.reason

            rows.append(row)
    return rows


def show_operation_trace(parameters: AdoShowTraceCommandParameters) -> None:
    """
    Show the measurement trace (requests and results) for an operation.

    This function provides a unified view of measurement requests and results,
    with support for filtering and multiple output formats.
    """
    # Select appropriate hidable fields based on view mode
    hidable_fields = (
        _RESULT_HIDABLE_FIELDS
        if parameters.unroll_entities
        else _REQUEST_HIDABLE_FIELDS
    )

    # Validate hide_fields parameter
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

    # Get SQL store (has its own spinner)
    sql_store = get_sql_store(parameters.ado_configuration.project_context)  # type: ignore[arg-type]

    # Fetch measurement requests with filters
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        try:
            # Get samplestore directly from operation
            samplestore = SampleStore.from_operation_identifier(
                operation_id=parameters.resource_id,
                metastore=sql_store,  # type: ignore[arg-type]
            )
        except (ResourceDoesNotExistError, NoRelatedResourcesError):
            status.stop()
            raise

        # Verify it's an SQLSampleStore to support filtering
        if not isinstance(samplestore, SQLSampleStore):
            console_print(
                f"{ERROR}This command requires an SQLSampleStore",
                stderr=True,
            )
            raise typer.Exit(1)

        status.update("Fetching measurements")

        # field_selectors are already prepared for DB by the command layer
        measurement_requests = samplestore.measurement_requests_for_operation(
            operation_id=parameters.resource_id,
            filters=parameters.field_selectors or None,
        )

    # Handle YAML output format
    if parameters.output_format == AdoShowTraceSupportedOutputFormats.YAML:
        # Output list of MeasurementRequests (which already include results)
        yaml_output = pydantic_model_as_yaml(measurement_requests)  # type: ignore[arg-type]

        # Write to file or stdout
        if parameters.output_file:
            parameters.output_file.write_text(yaml_output)
        else:
            console_print(yaml_output)
        return

    # Build rows based on view mode
    if parameters.unroll_entities:
        rows = _build_result_level_rows(measurement_requests)
    else:
        rows = _build_request_level_rows(measurement_requests)

    # Convert to DataFrame and apply column hiding
    import pandas as pd

    from orchestrator.utilities.pandas import reorder_dataframe_columns

    df = pd.DataFrame(rows)

    # Reorder columns to move metadata to the end
    if parameters.unroll_entities:
        df = reorder_dataframe_columns(
            df=df,
            move_to_start=[],
            move_to_end=[
                _RESULT_COLUMN.INVALID_REASON.value,
                _RESULT_COLUMN.REQUEST_METADATA.value,
                _RESULT_COLUMN.RESULT_METADATA.value,
            ],
        )
    else:
        df = reorder_dataframe_columns(
            df=df, move_to_start=[], move_to_end=[_REQUEST_COLUMN.METADATA.value]
        )

    if parameters.hide_fields:
        df = df.drop(parameters.hide_fields, axis="columns", errors="ignore")

    # Output the dataframe
    df_to_output(
        df=df,
        output_format=parameters.output_format.value,
        output_file=parameters.output_file,
        no_trunc=parameters.no_trunc,
    )


# Made with Bob
